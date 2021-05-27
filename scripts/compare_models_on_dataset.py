"""
Example Usage: 

python scripts/compare_models_on_dataset.py data/analysis_results/glue_mrpc_1+2_distilgpt2.h5 data/analysis_results/glue_mrpc_1+2_gpt2.h5 -o data/compared_results
"""

import argparse
from analysis.analysis_results_dataset import H5AnalysisResultDataset
from typing import *
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from analysis import LMAnalysisOutputH5, H5AnalysisResultDataset
from tqdm import tqdm
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument(
    "ds1",
    type=str,
    help="path to first H5AnalysisResultDataset",
)
parser.add_argument(
    "ds2",
    type=str,
    help="path to second h5 file",
)
parser.add_argument(
    "--output_f",
    "-o",
    type=str,
    default=None,
    help="Where to store the output h5 file. If not provided or is the name of existing directory, create using name of provided ds and model",
)
parser.add_argument(
    "--max_clamp_rank",
    type=int,
    default=50,
    help="Ranks beyond this are clamped to this value",
)

args = parser.parse_args()


def ex_compare(ex1: LMAnalysisOutputH5, ex2: LMAnalysisOutputH5, max_rank=50):
    r1 = ex1.ranks.astype(np.int32)
    r2 = ex2.ranks.astype(np.int32)
    clamped_r1 = np.clip(r1, 0, max_rank)
    clamped_r2 = np.clip(r2, 0, max_rank)
    p1 = ex1.probs
    p2 = ex2.probs

    rank_diff = r2 - r1
    prob_diff = p2 - p1
    clamped_rank_diff = clamped_r2 - clamped_r1
    kl_diff = F.kl_div(torch.tensor(p1), torch.tensor(p2), reduction="sum")

    topk_token_set1 = [set(t) for t in ex1.topk_token_ids]
    topk_token_set2 = [set(t) for t in ex2.topk_token_ids]
    n_topk_diff = np.array([len(s1.difference(s2)) for s1, s2 in zip(topk_token_set1, topk_token_set2)])

    return {
        "n_tokens": len(r1),
        "avg_rank_diff": np.mean(rank_diff),
        "max_rank_diff": np.max(rank_diff),
        "avg_clamped_rank_diff": np.mean(clamped_rank_diff),
        "max_clamped_rank_diff": np.max(clamped_rank_diff),
        "avg_prob_diff": np.mean(prob_diff),
        "max_prob_diff": np.max(prob_diff),
        "kl": kl_diff.item(),
        "avg_topk_diff": n_topk_diff.mean(),
        "max_topk_diff": n_topk_diff.max()
    }

# Smart defaults

ds1 = H5AnalysisResultDataset.from_file(args.ds1)
ds2 = H5AnalysisResultDataset.from_file(args.ds2)


assert ds1.dataset_name == ds2.dataset_name, "The two datasets should have the same name"
ds_name = ds1.dataset_name
assert ds1.dataset_checksum == ds2.dataset_checksum, "The two datasets should have the same checksum of contents"
assert ds1.vocab_hash == ds2.vocab_hash, "The two datasets should be created by models that share the same vocabulary"

default_name = f"{ds1.model_name}_{ds2.model_name}_{ds_name}.csv"
if args.output_f is None:
    output_name = Path(default_name)
elif Path(args.output_f).is_dir():
    output_name = Path(args.output_f) / default_name
else:
    output_name = Path(args.output_f)
output_f = Path(output_name)
output_f.parent.mkdir(parents=True, exist_ok=True)


diff_ab = [ex_compare(exa, exb, max_rank=args.max_clamp_rank)for exa, exb in tqdm(zip(ds1, ds2), total=len(ds1))]
df = pd.DataFrame(data=diff_ab)
print(f"Saving analsysis results to {output_f}")
df.to_csv(output_f)