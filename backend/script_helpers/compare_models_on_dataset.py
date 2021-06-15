import typer
from analysis.analysis_results_dataset import H5AnalysisResultDataset
from typing import *
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from analysis import LMAnalysisOutputH5, H5AnalysisResultDataset
from tqdm import tqdm
import pandas as pd
import path_fixes as pf

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
    # kl_diff = F.kl_div(torch.tensor(p1), torch.tensor(p2), reduction="sum")

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
        # "kl": kl_diff.item(),
        "avg_topk_diff": n_topk_diff.mean(),
        "max_topk_diff": n_topk_diff.max()
    }

def compare_datasets(ds1_name, ds2_name, output_dir, max_clamp_rank):
    ds1 = H5AnalysisResultDataset.from_file(ds1_name)
    ds2 = H5AnalysisResultDataset.from_file(ds2_name)


    assert ds1.dataset_name == ds2.dataset_name, "The two datasets should have the same name"
    ds_name = ds1.dataset_name
    assert ds1.dataset_checksum == ds2.dataset_checksum, "The two datasets should have the same checksum of contents"

    # Below is BROKEN because python's `hash` function changes between process runs
    assert ds1.vocab_hash == ds2.vocab_hash, "The two datasets should be created by models that share the same vocabulary"

    default_name = f"{ds1.model_name}_{ds2.model_name}_{ds_name}.csv"
    output_f = output_dir / default_name

    if output_f.exists():
        error = FileExistsError(f"Will not override existing {output_f}")
        error.details = {}
        error.details['outfname'] = str(output_f)
        raise error

    diff_ab = [ex_compare(exa, exb, max_rank=max_clamp_rank)for exa, exb in tqdm(zip(ds1, ds2), total=len(ds1))]
    df = pd.DataFrame(data=diff_ab)
    print(f"     Saving analysis results to {output_f}")
    df.to_csv(output_f)

    return output_f

def compare_models_on_dataset(
    ds1: str = typer.Argument(..., help="path to first H5AnalysisResultDataset"),
    ds2: str = typer.Argument(..., help="path to second H5AnalysisResultDataset"),
    output_dir: str = str(pf.COMPARISONS),
    max_clamp_rank: int = typer.Option(50, help="Ranks beyond this are clamped to this value"),
    invert: bool = typer.Option(True, help="Compute an ds1 -> ds2 evaluation in addition to an ds2 -> ds1 evaluation. Note that some of the 'metrics' are asymmetric"),
):
    """Calculate the difference between two comparable models evaluated on the same dataset

    Args:
        ds1 (str, optional): The path to the first `datasetXmodel.h5`.
        ds2 (str, optional): The path to the first `datasetXmodel.h5`.
        output_dir (str, optional): Where to save the output. Defaults to COMPARISONS directory for this application
        max_clamp_rank (int, optional): Ranks beyond this are clamped to this value.
        invert (bool, optional): Compute an ds1 -> ds2 evaluation in addition to an ds2 -> ds1 evaluation. Note that some of the 'metrics' are asymmetric.

    Raises:
        ValueError: Raised if the `output_dir` argument is an existing file.
    """
    output_dir = Path(output_dir)
    if output_dir.is_file(): raise ValueError("Specified output dir cannot be an existing file")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        output_f1 = compare_datasets(ds1, ds2, output_dir, max_clamp_rank)
    except FileExistsError as e:
        print(e)
        output_f1 = e.details['outfname']

    output_f2 = None
    if invert:
        print("\n\nRepeating with inverted datasets\n\n")
        try:
            output_f2 = compare_datasets(ds2, ds1, output_dir, max_clamp_rank)
        except FileExistsError as e:
            print(e)
            output_f2 = e.details['outfname']

    return output_f1, output_f2