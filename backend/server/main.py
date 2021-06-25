import argparse
from analysis.analysis_results_dataset import H5AnalysisResultDataset
from analysis.helpers import model_name2path, model_path2name
import os
import json
import re
import torch
import numpy as np
from enum import Enum
from pathlib import Path
from typing import *
from functools import lru_cache
import pandas as pd
from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import server.types as types
from server.utils import deepdict_to_json, SortOrder
from analysis import AutoLMPipeline, analyze_text
import path_fixes as pf

from api import LMComparer, ModelManager

__author__ = "DreamTeam V1.5: Hendrik Strobelt, Sebastian Gehrmann, Ben Hoover"


@lru_cache
def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", default="gpt-2-small")
    parser.add_argument("--address", default="127.0.0.1")  # 0.0.0.0 for nonlocal use
    parser.add_argument(
        "--port", type=int, default=8000, help="Port on which to run the app."
    )
    parser.add_argument("--dir", type=str, default=os.path.abspath("data"))
    parser.add_argument("--suggestions", type=str, default=os.path.abspath("data"))
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Use a custom analysis folder rather than the default paths to compare models",
    )
    parser.add_argument(
        "--tokenization-type",
        "-t",
        type=str,
        default="gpt",
        help="One of {'gpt', 'bert'}. Tells the frontend how to visualize the tokens used by the model.",
    )

    args, _ = parser.parse_known_args()
    return args


@dataclass
class ServerConfig:
    ANALYSIS: Union[Path, str]
    COMPARISONS: Union[Path, str]
    custom_dir: bool = False
    m1: Optional[str] = None  # Only available if 'custom_dir' is true
    m2: Optional[str] = None  # Only available if 'custom_dir' is true
    dataset: Optional[str] = None  # Only available if 'custom_dir' is true


@lru_cache
def get_config() -> ServerConfig:
    config_dir = get_args().config
    if config_dir is None:
        return ServerConfig(
            ANALYSIS=pf.ANALYSIS, COMPARISONS=pf.COMPARISONS, custom_dir=False
        )

    config_dir = Path(config_dir)

    with open(config_dir / "metadata.json", "r") as fp:
        metadata = json.load(fp)

    dataset = Path(metadata["dataset"]).stem

    return ServerConfig(
        ANALYSIS=config_dir,
        COMPARISONS=config_dir,
        custom_dir=True,
        m1=metadata["m1"],
        m2=metadata["m2"],
        dataset=dataset,
    )


@lru_cache
def get_pipeline(name: str):
    return AutoLMPipeline.from_pretrained(name)


@lru_cache
def get_comparison_results(m1: str, m2: str, dataset: str):
    m1_path_name = model_name2path(m1)
    m2_path_name = model_name2path(m2)
    results_fname = (
        get_config().COMPARISONS / f"{m1_path_name}_{m2_path_name}_{dataset}.csv"
    )
    compared_results = pd.read_csv(str(results_fname), index_col=0)
    return compared_results


@lru_cache
def get_analysis_results(dataset: str, mname: str):
    model_path_name = model_name2path(mname)
    return H5AnalysisResultDataset.from_file(
        str(get_config().ANALYSIS / f"{dataset}{pf.ANALYSIS_DELIM}{model_path_name}.h5")
    )


def list_all_datasets():
    return [p.stem.split(pf.ANALYSIS_DELIM) for p in get_config().ANALYSIS.glob("*.h5")]


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

lru = {}
model_manager = ModelManager()


class AvailableMetrics(str, Enum):
    avg_rank_diff = "avg_rank_diff"
    max_rank_diff = "max_rank_diff"
    avg_clamped_rank_diff = "avg_clamped_rank_diff"
    max_clamped_rank_diff = "max_clamped_rank_diff"
    avg_prob_diff = "avg_prob_diff"
    max_prob_diff = "max_prob_diff"
    # kl = "kl"
    avg_topk_diff = "avg_topk_diff"
    max_topk_diff = "max_topk_diff"


available_metrics = set(AvailableMetrics._member_names_)


# Main routes
@app.get("/")
def index():
    """For local development, serve the index.html in the dist folder"""
    return RedirectResponse(url="client/index.html")


# the `file_path:path` says to accept any path as a string here. Otherwise, `file_paths` containing `/` will not be served properly
@app.get("/client/{file_path:path}")
def send_static_client(file_path: str):
    """Serves (makes accessible) all files from ./client/ to ``/client/{path}``. Used primarily for development. NGINX handles production.

    Args:
        path: Name of file in the client directory
    """
    f = str(pf.DIST / file_path)
    print("Finding file: ", f)
    return FileResponse(f)


@app.get("/data/{path:path}")
def send_data(path):
    """serves all files from the data dir to ``/data/{path:path}``

    Args:
        path: Path from api call
    """
    f = Path(get_args().dir) / path
    print("Finding data file: ", f)
    return FileResponse(f)


# ======================================================================
## MAIN API ##
# ======================================================================
@app.get("/api/available-datasets")
def get_available_datasets(m1: str, m2: str):
    if get_config().custom_dir:
        return [get_config().dataset]

    if m1 == "" or m2 == "":
        return []

    h5paths = list_all_datasets()
    m1name = model_name2path(m1)
    m2name = model_name2path(m2)
    m1_h5 = set([p[0] for p in h5paths if p[1] == m1name])
    m2_h5 = set([p[0] for p in h5paths if p[1] == m2name])
    available_datasets = list(m1_h5.intersection(m2_h5))
    return available_datasets


@app.get("/api/all-models")
def get_all_models():
    if get_config().custom_dir:
        return [
            {
                "model": get_config().m1,
                "type": "custom",
                "token": get_args().tokenization_type,
            },
            {
                "model": get_config().m2,
                "type": "custom",
                "token": get_args().tokenization_type,
            },
        ]

# ☘🍀🌼🌻🌺🌹💐🌸

    res = [
        {"model": "gpt2", "type": "🍀", "token": "gpt"},
        {"model": "distilgpt2", "type": "🍀", "token": "gpt"},
        {"model": "lysandre/arxiv-nlp", "type": "🍀", "token": "gpt"},
        {"model": "bert-base-uncased", "type": "🌼", "token": "bert"},
        {"model": "distilbert-base-uncased", "type": "🌼", "token": "bert"},
        {"model": "distilbert-base-uncased-finetuned-sst-2-english", "type": "🌼", "token": "bert"},
        {"model": "bert-base-cased", "type": "🌺", "token": "bert"},
        {"model": "distilbert-base-cased", "type": "🌺", "token": "bert"},

        # {"model": "lysandre/arxiv", "type": "gpt", "token": "gpt"},
    ]

    return res


@app.post("/api/specific-attention")
def specific_attention(payload: types.SpecificAttentionRequest):
    pp1 = get_pipeline(payload.m1)
    pp2 = get_pipeline(payload.m2)

    output1 = pp1.forward(payload.text)
    output2 = pp2.forward(payload.text)

    att1 = output1.attentions
    att2 = output2.attentions

    idx = payload.token_index_in_text
    if payload.outward_attentions:
        a1 = att1[:, :, idx, :]
        a2 = att2[:, :, idx, :]
    else:
        a1 = att1[:, :, :, idx]
        a2 = att2[:, :, :, idx]

    return {
        "m1": deepdict_to_json(a1, ndigits=4, force_float=True),
        "m2": deepdict_to_json(a2, ndigits=4, force_float=True),
    }


@app.get("/api/new-suggestions")
def new_suggestions(
    m1: str,
    m2: str,
    dataset: str,
    metric: AvailableMetrics,
    order: SortOrder = "descending",
    k: int = 50,
    sort_by_abs: bool = True,
    histogram_bins:int = 100,
):
    f"""Get the comparison between model m1 and m2 on the dataset. Rank the output according to a valid metric

    Args:
        m1 (str): The name of the first model to compare
        m2 (str): The name of the second model to compare. Should have the same tokenizer as m1
        dataset (str): The name of the dataset both the models already analyzed.
        metric (str): One of the available metrics: '{available_metrics}'
        order (SortOrder, optional): If "ascending", sort in order of least to greatest. Defaults to "descending".
        k (int, optional): The number of interesting instances to return. Defaults to 50.
        sort_by_abs (bool): Sort by the absolute value. Defaults to True
        histogram_bins (int): Number of bins in the returned histogram of all values. Defaults to 100.

    Returns:
        Object containing information to statically analyze two models.
    """
    pp1 = get_pipeline(m1)
    pp2 = get_pipeline(m2)
    ds1 = get_analysis_results(dataset, m1)
    ds2 = get_analysis_results(dataset, m2)
    compared_results = get_comparison_results(m1, m2, dataset)
    results = np.array(compared_results[metric])

    if sort_by_abs:
        results_sign = np.sign(results)
        results_abs = np.abs(results)

        sign = -1 if order == "descending" else 1
        sort_idxs = np.argsort(sign * results_abs)[:k]
        df_sorted = compared_results.iloc[sort_idxs]
        results_sign = results_sign[sort_idxs]

    else:
        df_sorted = compared_results.sort_values(
            by=metric, ascending=("ascending" == order)
        )[:k]
        results_sign = np.ones(len(df_sorted))

    metrics = df_sorted.to_dict(orient="index")

    def proc_data_row(x1, x2):
        tokens = pp1.tokenizer.convert_ids_to_tokens(x1.token_ids)
        text = x1.phrase
        m1_info = {
            "rank": x1.ranks,
            "prob": x1.probs,
            "topk": pp1.idmat2tokens(torch.tensor(x1.topk_token_ids)),
        }
        m2_info = {
            "rank": x2.ranks,
            "prob": x2.probs,
            "topk": pp2.idmat2tokens(torch.tensor(x2.topk_token_ids)),
        }
        diff = {"rank": x2.ranks - x1.ranks, "prob": x2.probs - x1.probs}
        return {
            "tokens": tokens,
            "text": text,
            "m1": m1_info,
            "m2": m2_info,
            "diff": diff,
        }

    result = [
        deepdict_to_json(o, ndigits=3)
        for o in [
            dict(
                {"example_idx": k, "metrics": v, "sign": results_sign[i]},
                **proc_data_row(ds1[k], ds2[k]),
            )
            for i, (k, v) in enumerate(metrics.items())
        ]
    ]

    histogram_values, bin_edges = np.histogram(results, bins=histogram_bins)

    return {
        "request": {
            "m1": m1,
            "m2": m2,
            "dataset": dataset,
            "metric": metric,
            "order": order,
            "k": k,
        },
        "result": result,
        "histogram": deepdict_to_json({
            "values": histogram_values,
            "bin_edges": bin_edges
        }, ndigits=4)
    }


@app.post("/api/analyze-text")
def analyze_models_on_text(payload: types.AnalyzeRequest):
    m1 = payload.m1
    m2 = payload.m2
    pp1 = get_pipeline(m1)
    pp2 = get_pipeline(m2)
    text = payload.text
    output = analyze_text(text, pp1, pp2)
    result = deepdict_to_json(output, ndigits=4)

    res = {"request": {"m1": m1, "m2": m2, "text": text}, "result": result}
    return res


@app.post("/api/analyze")
def analyze(payload: types.AnalyzeRequest):
    m1 = payload.m1
    m2 = payload.m2
    text = payload.text

    # TODO: hacky cache
    c_key = str(m1) + str(m2) + text
    if c_key in lru:
        return lru[c_key]

    model1, tok1 = model_manager.get_model_and_tokenizer(m1)

    model2, tok2 = None, None
    if m2:
        model2, tok2 = model_manager.get_model_and_tokenizer(m2)

    comparer = LMComparer(model1, model2, tok1, tok2)
    res = comparer.analyze_text(text)

    res = {"request": {"m1": m1, "m2": m2, "text": text}, "result": res}
    lru[c_key] = res

    return res


if __name__ == "__main__":
    # This file is not run as __main__ in the uvicorn environment
    # args, _ = parser.parse_known_args()
    args = get_args()
    uvicorn.run("server:app", host=args.address, port=args.port)
