from typing import *
import pickle
import numpy as np
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    AutoModelWithLMHead,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from hashlib import sha256
from .helpers import topk_token_diff, LMAnalysisOutput, LMAnalysisOutputH5

import torch
import h5py
from dataclasses import dataclass


def get_group(f: Union[h5py.File, h5py.Group], gname: str):
    if gname in f.keys():
        return f[gname]
    return f.create_group(gname)


# !! Monkey patching
h5py.File.get_group = get_group
h5py.Group.get_group = get_group


def list2consistent_hash(lst):
    """Convert a sortable, shallow list to a consistent hash"""
    bstr = pickle.dumps(sorted(lst))
    return sha256(bstr).hexdigest()


def reduce_logits(logits):
    """Convert 3D logits (where each example represents a diff masked token) to 2D logits"""
    assert (
        logits.ndim == 3
    ), "Expected logits to be 3 dimensional where the first dimension represents the MASK sliding across each non-special token in the input"
    lgts2 = logits[0].clone()  # Without any masks
    for i in np.arange(1, logits.shape[0] - 1):
        lgts2[i, :] = logits[i, i, :]

    return lgts2


def reduce_attentions(attention, i: int):
    """Convert 4D attention to 3D attention for a layer's attentions

    `i` indicates which index is masked at the provided attention

    Only care about attentions OUT OF each MASKed token
    """
    att2 = attention[0].clone()

    # Editing OUTWARD attentions
    for i in np.arange(1, attention.shape[0] - 1):
        att2[:, i, :] = attention[i, :, i, :]  # MASK_i, heads, MASK_i, outward atts
    return att2


class AnalysisLMPipelineForwardOutput(CausalLMOutputWithCrossAttentions):
    def __init__(self, phrase: str, token_ids: torch.tensor, **kwargs):
        super().__init__(**kwargs)
        self.N = len(token_ids)
        self.token_ids = token_ids
        self.phrase = phrase


class AutoLMPipeline:
    def __init__(self, model, tokenizer):

        self.model: transformers.PreTrainedModel = model
        self.device = self.model.device
        self.tokenizer: transformers.PreTrainedTokenizer = tokenizer
        self.config = self.model.config
        self.vocab_hash = list2consistent_hash(self.tokenizer.vocab.items())

        # Only one should be true below:
        self.is_auto_regressive = "gpt" in self.model.config.model_type
        self.is_maskable = 'bert' in self.model.config.model_type

        assert self.is_auto_regressive or self.is_maskable, "Needs to be a gpt- or bert-like model"

        self.tokenizer.pad_token = (
            self.tokenizer.eos_token
            if self.is_auto_regressive
            else self.tokenizer.pad_token
        )
        self.missing_special_tokens = len(self.tokenizer) != self.model.config.vocab_size

    @classmethod
    def from_pretrained(cls, name_or_path, device = None):
        """Create a model and tokenizer from a single name, passing arguemnts to `from_pretrained` of the AutoTokenizer and AutoModel"""
        model = AutoModelWithLMHead.from_pretrained(name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        if device is None and torch.cuda.is_available():
            device = 0
            model = model.to(device)
        else:
            model = model.to(device)
        return cls(model, tokenizer)

    def for_model(self, s):
        tids = self.tokenizer.encode(s, return_tensors="pt").to(self.device)

        if self.is_auto_regressive:
            # tids = self.for_model(self.tokenizer.bos_token + s)
            add_tok = 0 if self.missing_special_tokens else self.tokenizer.bos_token_id
            tids = torch.cat([torch.tensor([add_tok], device=self.device), tids.squeeze()])
            return tids
        elif self.is_maskable:
            N = len(tids[0])

            # Assume CLS and SEP
            N_no_special = N - 2

            # Mask every word (Except CLS and SEP) and treat that input as part of a batch
            # First input is unmodified
            tid2 = tids.repeat((N_no_special + 1, 1))
            rows, cols = range(1, N_no_special), range(1, N_no_special)
            tid2[rows, cols] = self.tokenizer.mask_token_id
            return tid2
        else:
            raise NotImplementedError(f"Model type is not autoregressive or maskable.")

    def forward(self, s, output_attentions=False) -> AnalysisLMPipelineForwardOutput:
        with torch.no_grad():
            tids = self.for_model(s)
            output = self.model(tids, output_attentions=output_attentions)

            if self.is_auto_regressive:
                output.logits = output.logits.squeeze()
                # Offset this as well
                tids = tids.squeeze()
                output.logits = output.logits[:-1]
                tids = tids[1:]

                # SWAP THE COMMENTS ON THE BELOW 2 LINES IF WE PREFER THE OTHER FUNCTIONALITY
                if output_attentions:
                    output.attentions = torch.cat(
                        [a[:, :, 1:, 1:] for a in output.attentions], dim=0
                    )
                else:
                    output.attentions = None

            elif self.is_maskable:

                output.logits = reduce_logits(output.logits)
                tids = tids[0]  # Where everything is unmasked
                
                if output_attentions:
                    output.attentions = torch.cat(
                        [
                            reduce_attentions(a, i).unsqueeze(0)
                            for i, a in enumerate(output.attentions)
                        ],
                        dim=0,
                    )
                else:
                    output.attentions = None

                # Remove CLS and SEP
                output.logits = output.logits[1:-1]
                if output_attentions: 
                    output.attentions = output.attentions[:, :, 1:-1, 1:-1]
                tids = tids[1:-1]
            else:
                raise ValueError(
                    "Unhandled model type. Model type is not autoregressive or maskable"
                )

            new_output = AnalysisLMPipelineForwardOutput(s, tids, **output.__dict__)

        return new_output

    def idmat2tokens(self, idmat: torch.tensor) -> List[List[str]]:
        """Convert arbitrarily nested IDs into tokens"""
        output = []
        for i, idlist in enumerate(idmat):
            if idlist.ndim == 1:
                output.append(self.tokenizer.convert_ids_to_tokens(idlist))
        return output


def collect_analysis_info(
    model_output: AnalysisLMPipelineForwardOutput, k=10
) -> LMAnalysisOutput:
    """
    Analyze the output of a language model for probabilities and ranks

    Args:
        model_output: The output of the causal Language Model
        k: The number of top probabilities we care about
    """
    probs = torch.softmax(model_output.logits, dim=1)
    ranks = (
        torch.argsort(model_output.logits, dim=1, descending=True)
        == model_output.token_ids.unsqueeze(1).expand_as(model_output.logits)
    ).nonzero()[:, 1]
    phrase_probs = probs[torch.arange(model_output.N), model_output.token_ids]
    #     topk_logit_values, topk_logit_inds = torch.topk(output.logits, k=k, dim=1)
    topk_prob_values, topk_prob_inds = torch.topk(probs, k=k, dim=1)
    attention = model_output.attentions  # Layer, Head, N, N (or None)

    return LMAnalysisOutput(
        phrase=model_output.phrase,
        token_ids=model_output.token_ids,
        ranks=ranks,
        probs=phrase_probs,
        topk_prob_values=topk_prob_values,
        topk_token_ids=topk_prob_inds,
        attention=attention,
    )


def zipTopK(tokens_topk: List[List[str]], probs):
    res = []
    for i, topks in enumerate(tokens_topk):
        res.append(list(zip(topks, probs[i].tolist())))

    return res


def analyze_text(text: str, pp1: AutoLMPipeline, pp2: AutoLMPipeline, topk=10):
    assert (
        pp1.vocab_hash == pp2.vocab_hash
    ), "Vocabularies of the two pipelines must align"

    output1 = pp1.forward(text)
    output2 = pp2.forward(text)

    parsed_output1 = collect_analysis_info(output1, k=topk)
    parsed_output2 = collect_analysis_info(output2, k=topk)

    token_ids = parsed_output1.token_ids
    tokens = pp1.tokenizer.convert_ids_to_tokens(token_ids)

    def clamp(arr, max_rank=50):
        return np.clip(arr, 0, max_rank)

    return {
        "text": text,
        "tokens": tokens,
        "m1": {
            "rank": parsed_output1.ranks,
            "prob": parsed_output1.probs,
            "topk": zipTopK(
                pp1.idmat2tokens(parsed_output1.topk_token_ids),
                parsed_output1.topk_prob_values,
            ),  # Turn to str
        },
        "m2": {
            "rank": parsed_output2.ranks,
            "prob": parsed_output2.probs,
            "topk": zipTopK(
                pp2.idmat2tokens(parsed_output2.topk_token_ids),
                parsed_output2.topk_prob_values,
            ),  # Turn to str
        },
        "diff": {
            "rank": parsed_output2.ranks - parsed_output1.ranks,
            "prob": parsed_output2.probs - parsed_output1.probs,
            "rank_clamp": clamp(parsed_output2.ranks.cpu())
            - clamp(parsed_output1.ranks.cpu()),
            "topk": topk_token_diff(
                parsed_output1.topk_token_ids.tolist(),
                parsed_output2.topk_token_ids.tolist(),
            ),
        },
    }
