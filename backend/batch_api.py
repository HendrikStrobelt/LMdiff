import numpy as np
import torch
from torch.nn import functional as F
from transformers import (
    AutoTokenizer,
    AutoModelWithLMHead,
    PreTrainedModel,
    PreTrainedTokenizer,
    GPT2Tokenizer, GPT2LMHeadModel)
from typing import *

class ModelManager:
    def __init__(self):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.tokenizers = {}

    def get_model_and_tokenizer(self, model_name: str):
        model = self.models.get(model_name, None)
        tokenizer = self.tokenizers.get(model_name, None)
        if (model is not None) and (tokenizer is not None):
            return model, tokenizer
        elif model_name.find('arxiv') >= 0:
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            model = GPT2LMHeadModel.from_pretrained(model_name)\
                .to(self.device)
            return model, tokenizer
        else:
            model = AutoModelWithLMHead.from_pretrained( model_name).to(self.device)
            print(f"Model is using {self.device}")
            self.models[model_name] = model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizers[model_name] = tokenizer
            return model, tokenizer


def format_attn(attention_tuples: tuple):
    """
    Input: N tuples (N = layer num)

    Each tuple item is Tensor of shape
    Batch x num heads x from x to

    Output: Tensor of shape layer x from x to
    (averaged over heads)
    """

    # Combine tuples into large Tensor, then avg
    return torch.cat([l for l in attention_tuples], dim=0).mean(dim=1)


class LMComparer:
    def __init__(
            self,
            m1: PreTrainedModel,
            m2: PreTrainedModel,
            t1: PreTrainedTokenizer,
            t2: PreTrainedTokenizer,
    ):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.m1 = m1
        self.m1.eval()
        self.m2 = m2
        self.m2.eval()

        self.tokenizer = t1
        self.bos_token_id = self.tokenizer.bos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Check that both use same tokenizer
        assert type(self.tokenizer) == type(
            t2
        ), "Please use models with same tokenization scheme"

    def get_rank_prob_topk( self, y: torch.Tensor, probs: torch.Tensor, k: int = 5):
        """
        Args:
            y: IDs of the tokenized input (no generation token at the beginning)
            probs: Probabilities of every token in the vocabulary at that position
            tokenizer: Tokenizer that generated the probabilities
            k: how many top tokens to report

        Returns:
            Payload containing information needed for diffing language models
        """
        # Vocabulary sorted by logits
        top_voc = torch.argsort(probs, descending=True)
        
        # Returning `as_tuple=True` allows indexing with output
        yrank_idx = torch.eq(y.unsqueeze(-1), top_voc).nonzero(as_tuple=True)
        
        # Assigning ranks to each input_id
        yranks = torch.zeros_like(y)
        yranks[yrank_idx[:2]] = yrank_idx[-1]
        
        # Probabilities of actual inputs
        yrank_idx_og = (yrank_idx[0], yrank_idx[1], top_voc[yrank_idx])
        yprobs = probs[yrank_idx_og].view(y.shape) # TODO: CHECK that reshape is correctly done
        
        topk = top_voc[:, :, :k]

        # I expect this list comprehension to be pretty slow. Should maybe do once at the end?
        topk_words = [[self.tokenizer.convert_ids_to_tokens(preds) for preds in sentence] for sentence in topk]
        return yranks, yprobs, topk_words

    def batch_forward(self, text: List[str], k: int = 7):
        """Batched processing of all the information needed to analyze a language model
        
        Args:
            text: Sentence batch to analyze
            k: How many predictions we care to analyze

        Returns:
            Payload containing information needed to diff models
        """
        encoded = self.tokenizer.batch_encode_plus(text, pad_to_max_length=True, return_tensors="pt")
        ids = encoded["input_ids"]
        start_token = self.bos_token_id
        start_tokens = (torch.ones(ids.shape[0], dtype=torch.int64) * start_token).view((-1, 1))

        start_1s = torch.ones((ids.shape[0], 1), dtype=torch.int64)
        gen_ids = torch.cat((start_tokens, ids), dim=1)

        # Start all inputs with GPT2's EOS token
        encoded['input_ids'] = gen_ids

        # Allow attention to EOS token
        encoded['attention_mask'] = torch.cat((start_1s, encoded['attention_mask']), dim=1)

        m1_logits, m1_embeds, atts1 = self.m1(**encoded, output_attentions=True)
        m2_logits, m2_embeds, atts2 = self.m2(**encoded, output_attentions=True)

        attn1 = format_attn(atts1)
        attn2 = format_attn(atts2)

        probs1 = F.softmax(m1_logits[:, :-1], dim=-1)
        probs2 = F.softmax(m2_logits[:, :-1], dim=-1)
        assert probs1.shape == probs2.shape, "Vocab sizes not the same"

        ranks1, probs1, topk_words1 = self.get_rank_prob_topk(ids, probs1, k)
        ranks2, probs2, topk_words2 = self.get_rank_prob_topk(ids, probs2, k)
        
        rank_diff = ranks2 - ranks1
        probs_diff = probs2 - probs1
        attn_diff = attn2 - attn1 if attn1.shape == attn2.shape else None

        kl = F.kl_div(probs1, probs2, reduction="none") # Elementwise KL Div

        return {
            "prob": {
                "m1": probs1,
                "m2": probs2,
                "diff": probs_diff
            },
            "rank": {
                "m1": ranks1,
                "m2": ranks2,
                "diff": rank_diff
            },
            "topk": {
                "m1": topk_words1,
                "m2": topk_words2
            },
            "attn": {
                "m1": attn1,
                "m2": attn2,
                "diff": attn_diff
            },
            "kl": kl,
            "ids": ids,
            "tokens": [self.tokenizer.convert_ids_to_tokens(id) for id in ids],
            "text": text,
            "attention_mask": encoded['attention_mask']
        }
        

    def __call__(self, text: Union[List[str], str], k: int=7):
        """Handle single inputs or batched inputs"""
        if type(text) == str:
            return self.batch_forward([text], k)
        
        return self.batch_forward(text)


if __name__ == "__main__":
    mm = ModelManager()
    m1, t1 = mm.get_model_and_tokenizer("gpt2")
    m2, t2 = mm.get_model_and_tokenizer("distilgpt2")

    # Example of how to run comparison of models
    comparer = LMComparer(m1, m2, t1, t2)
    print("loading successful!")
    comparer("this is a test of a single sentence!")
    comparer(["this is a test!", "and this is yet another test for the books!", "yeah dude"])
    print("checking successful!")
