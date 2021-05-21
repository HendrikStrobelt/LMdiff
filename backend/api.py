import numpy as np
import torch
from torch.nn import functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    GPT2TokenizerFast, GPT2LMHeadModel)


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
            tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
            model = GPT2LMHeadModel.from_pretrained(model_name,
                                                    output_attentions=True)\
                .to(self.device)
            return model, tokenizer
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, output_attentions=True
            ).to(self.device)
            print(f"Model is using {self.device}")
            self.models[model_name] = model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizers[model_name] = tokenizer
            return model, tokenizer


def get_rank_prob_and_topk(
        y: np.ndarray, yhat: torch.Tensor, tokenizer: PreTrainedTokenizer,
        topk: int = 5
):
    # Sort the prob distribution
    sorted_preds = np.argsort(-yhat.data.cpu().numpy())
    # Get the rank of correct pred within prediction
    y_rank = list(
        [int(np.where(sorted_preds[i] == y[i].item())[0][0]) for i in
         range(y.shape[0])]
    )
    # Get the prob of correct pred
    y_prob = yhat[np.arange(0, y.shape[0], 1), y].data.cpu().numpy().tolist()

    pred_topk = [
        list(
            zip(
                [
                    tokenizer.convert_ids_to_tokens(int(p))
                    for p in sorted_preds[i][:topk]
                ],
                list(
                    map(
                        lambda x: round(x, 5),
                        yhat[i][
                            sorted_preds[i][:topk]].data.cpu().numpy().tolist(),
                    )
                ),
            )
        )
        for i in range(y.shape[0])
    ]

    return y_rank, y_prob, pred_topk


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

        # Check that both use same tokenizer
        assert type(self.tokenizer) == type(
            t2
        ), "Please use models with same tokenization scheme"

    def analyze_text(self, text: str, topk: int = 7):
        # Process Input
        toked = self.tokenizer.encode(text)
        start_token = torch.full(
            (1, 1), self.bos_token_id, device=self.device, dtype=torch.long,
        )
        context = torch.tensor(toked, device=self.device,
                               dtype=torch.long).unsqueeze(0)
        context = torch.cat([start_token, context], dim=1)

        # Target for LM
        y = context.squeeze(0)[1:].cpu().numpy()

        # Softmax and prune last output
        m1_output = self.m1(context)
        m2_output = self.m2(context)
        probs1 = F.softmax(m1_output[0][:, :-1], dim=-1).squeeze(0)
        probs2 = F.softmax(m2_output[0][:, :-1], dim=-1).squeeze(0)
        assert probs1.shape == probs2.shape

        # Attention Extraction
        m1_attn = format_attn(m1_output[2])
        m2_attn = format_attn(m2_output[2])
        # Diff only works with same architecture
        if m1_attn.shape == m2_attn.shape:
            attn_diff = list(
                (m2_attn - m1_attn).cpu().detach().numpy().tolist())
        else:
            attn_diff = None

        # Rank and Prob computation
        y1_rank, y1_prob, m1_topk = get_rank_prob_and_topk(
            y, probs1, self.tokenizer, topk
        )
        y2_rank, y2_prob, m2_topk = get_rank_prob_and_topk(
            y, probs2, self.tokenizer, topk
        )

        def clamp(arr, max_rank=50):
            return np.clip(arr, 0, max_rank)

        # probability diff
        diff = list(np.array(y2_prob) - np.array(y1_prob))
        rank_diff = list((np.array(y2_rank) - np.array(y1_rank)).tolist())
        rank_diff_clamped = list((clamp(np.array(y2_rank))
                                  - clamp(np.array(y1_rank))).tolist())
        # KLs
        kl = [
            float(F.kl_div(p1, p2, reduction="sum").item())
            for p1, p2 in zip(probs1, probs2)
        ]

        prob_payload = {
            "prob_m1": y1_prob,
            "prob_m2": y2_prob,
            "rank_m1": y1_rank,
            "rank_m2": y2_rank,
            "rank_diff": rank_diff,
            "rank_diff_clamped": rank_diff_clamped,
            "topk_m1": m1_topk,
            "topk_m2": m2_topk,
            "kl": kl,
            "diff": diff,
        }

        attn_payload = {
            "attn_m1": list(m1_attn.cpu().detach().numpy().tolist()),
            "attn_m2": list(m2_attn.cpu().detach().numpy().tolist()),
            "diff": attn_diff,
        }

        payload = {}
        payload["tokens"] = self.tokenizer.convert_ids_to_tokens(context[0][1:])
        payload["prob"] = prob_payload
        payload["text"] = text

        return payload


if __name__ == "__main__":
    models = ModelManager()
    m1, t1 = models.get_model_and_tokenizer("gpt2")
    m2, t2 = models.get_model_and_tokenizer("distilgpt2")
    # Load Models
    comparer = LMComparer(m1, m2, t1, t2)
    print("loading successful!")
    comparer.analyze_text("this is a test!")
    print("checking successful!")
