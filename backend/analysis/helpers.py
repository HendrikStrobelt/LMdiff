import numpy as np
import torch
import h5py
from dataclasses import dataclass
from typing import *


@dataclass
class LMAnalysisOutputH5:
    token_ids: np.array  # N
    ranks: np.array  # N,
    probs: np.array  # N,
    topk_prob_values: np.array  # k, N
    topk_token_ids: np.array  # k, N
    attention: np.array  # Layer, Head, N, N
    phrase: str

    # Add attributes

    @classmethod
    def from_group(cls, grp):
        return cls(
            token_ids=np.array(grp["token_ids"]),
            ranks=np.array(grp["ranks"]),
            probs=np.array(grp["probs"]),
            topk_prob_values=np.array(grp["topk_probs"]),
            topk_token_ids=np.array(grp["topk_token_ids"]),
            attention=np.array(grp["attention"]),
            phrase=grp.attrs["phrase"],
        )

    def save_to_h5group(self, h5group: h5py.Group):
        h5group.create_dataset("token_ids", data=self.token_ids)
        h5group.create_dataset("ranks", data=self.ranks)
        h5group.create_dataset("probs", data=self.probs)
        h5group.create_dataset("topk_probs", data=self.topk_prob_values)
        h5group.create_dataset("topk_token_ids", data=self.topk_token_ids)
        h5group.create_dataset("attention", data=self.attention)
        h5group.attrs["phrase"] = self.phrase


@dataclass
class LMAnalysisOutput:
    token_ids: torch.tensor  # N,
    ranks: torch.tensor  # N,
    probs: torch.tensor  # N,
    topk_prob_values: torch.tensor  # k, N
    topk_token_ids: torch.tensor  # k, N
    attention: torch.tensor  # Layer, Head, N, N
    phrase: str

    def for_h5(self):
        return LMAnalysisOutputH5(
            token_ids=self.token_ids.cpu().numpy().astype(np.int64),
            ranks=self.ranks.cpu().numpy().astype(np.uint32),
            probs=self.probs.cpu().numpy().astype(np.float32),
            topk_prob_values=self.topk_prob_values.cpu().numpy().astype(np.float32),
            topk_token_ids=self.topk_token_ids.cpu().numpy().astype(np.int64),
            attention=self.attention.cpu().numpy().astype(np.float32),
            phrase=self.phrase,
        )

    def save_to_h5group(self, h5group: h5py.Group):
        self.for_h5().save_to_h5group(h5group)
        return h5group


def topk_token_diff(t1: List[List[int]], t2: List[List[int]]):
    topk_token_set1 = [set(t) for t in t1]
    topk_token_set2 = [set(t) for t in t2]
    n_topk_diff = np.array(
        [len(s1.difference(s2)) for s1, s2 in zip(topk_token_set1, topk_token_set2)]
    )
    return n_topk_diff


SLASH_REPLACE = "__SLASH__"


def model_name2path(name: str) -> str:
    """Convert model name to a name that can be stored in the filesystem

    Needed because huggingface uses `/` in their model names
    """
    return name.replace("/", SLASH_REPLACE)


def model_path2name(path: str) -> str:
    """Convert model path to the huggingface pretrained model name

    Needed because huggingface uses `/` in their model names
    """
    return path.replace(SLASH_REPLACE, "/")
