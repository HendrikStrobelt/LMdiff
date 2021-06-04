import argparse
from analysis.analysis_results_dataset import H5AnalysisResultDataset
import os
import json
import re
import torch
from enum import Enum
from pathlib import Path
from typing import *
from functools import lru_cache
import pandas as pd

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import server.types as types
from server.utils import deepdict_to_json, SortOrder
from analysis import AutoLMPipeline, analyze_text
import path_fixes as pf

from api import LMComparer, ModelManager

__author__ = "DreamTeam V1.5: Hendrik Strobelt, Sebastian Gehrmann, Ben Hoover"


@lru_cache
def get_pipeline(name: str):
    return AutoLMPipeline.from_pretrained(name)


@lru_cache
def get_comparison_results(m1: str, m2: str, dataset: str):
    results_fname = pf.COMPARISONS / f"{m1}_{m2}_{dataset}.csv"
    compared_results = pd.read_csv(str(results_fname), index_col=0)
    return compared_results


@lru_cache
def get_analysis_results(dataset: str, mname: str):
    return H5AnalysisResultDataset.from_file(str(pf.ANALYSIS / f"{dataset}{pf.ANALYSIS_DELIM}{mname}.h5"))


@lru_cache
def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", default="gpt-2-small")
    parser.add_argument("--address", default="127.0.0.1")  # 0.0.0.0 for nonlocal use
    parser.add_argument(
        "--port", type=int, default=5001, help="Port on which to run the app."
    )
    parser.add_argument("--dir", type=str, default=os.path.abspath("data"))
    parser.add_argument("--suggestions", type=str, default=os.path.abspath("data"))

    args, _ = parser.parse_known_args()
    return args

@lru_cache
def list_all_datasets():
    return [p.stem.split(pf.ANALYSIS_DELIM) for p in pf.ANALYSIS.glob("*.h5")]

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
    kl = "kl"
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
def get_available_datasets(m1: str, m2:str):
    if m1 == '' or m2 == '':
        return []

    h5paths = list_all_datasets()
    m1_h5 = set([p[0] for p in h5paths if p[1] == m1])
    m2_h5 = set([p[0] for p in h5paths if p[1] == m2])
    available_datasets = list(m1_h5.intersection(m2_h5))
    return available_datasets

@app.get("/api/all-models")
def get_all_models():
    res = [
        {"model": "gpt2"},
        {"model": "distilgpt2"},
        # {"model": "lysandre/arxiv-nlp"},
        # {"model": "lysandre/arxiv"},
    ]

    return res

@app.post("/api/specific-attention")
def specific_attention(payload:types.SpecificAttentionRequest):
    pp1 = get_pipeline(payload.m1)
    pp2 = get_pipeline(payload.m2)

    output1 = pp1.forward(payload.text)
    output2 = pp2.forward(payload.text)

    att1 = torch.cat(output1.attentions)
    att2 = torch.cat(output2.attentions)

    idx = payload.token_index_in_text
    if payload.outward_attentions:
        a1 = att1[:,:,idx, :]
        a2 = att2[:,:,idx, :]
    else:
        a1 = att1[:,:,:, idx]
        a2 = att2[:,:,:, idx]

    return {
        "m1": deepdict_to_json(a1, ndigits=4, force_float=True),
        "m2": deepdict_to_json(a2, ndigits=4, force_float=True)
    }

@app.get("/api/new-suggestions")
def new_suggestions(
    m1: str,
    m2: str,
    dataset: str,
    metric: AvailableMetrics,
    order: SortOrder = "descending",
    k: int = 50,
):
    f"""Get the comparison between model m1 and m2 on the dataset. Rank the output according to a valid metric

    Args:
        m1 (str): The name of the first model to compare
        m2 (str): The name of the second model to compare. Should have the same tokenizer as m1
        dataset (str): The name of the dataset both the models already analyzed.
        metric (str): One of the available metrics: '{available_metrics}'
        order (SortOrder, optional): If "ascending", sort in order of least to greatest. Defaults to "descending".
        k (int, optional): The number of interesting instances to return. Defaults to 50.

    Returns:
        Object containing information to statically analyze two models.
    """
    pp1 = get_pipeline(m1)
    pp2 = get_pipeline(m2)
    ds1 = get_analysis_results(dataset, m1)
    ds2 = get_analysis_results(dataset, m2)
    compared_results = get_comparison_results(m1, m2, dataset)

    df_sorted = compared_results.sort_values(
        by=metric, ascending=("ascending" == order)
    )[:k]
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
            dict({"example_idx": k, "metrics": v}, **proc_data_row(ds1[k], ds2[k]))
            for k, v in metrics.items()
        ]
    ]

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
    }


@app.get("/api/suggestions")
def get_suggestions(m1: str, m2: str, corpus: str = "wiki_split"):
    # corpus = suggestion.get('corpus', 'wiki_split')
    # suggestion['corpus'] = corpus
    # print(corpus)
    inverse_order = False
    base_path = Path(get_args().suggestions)

    models = [m1, m2]
    if m1 > m2:
        models = [m2, m1]
        inverse_order = True
    names = sorted([re.sub(r"[/\-]", "_", n) for n in models])
    json_name = "__".join(names) + "__phrases.json"
    filename = base_path / corpus / json_name

    if os.path.exists(filename):
        res = json.load(open(filename, "r"))
        res["inverse_order"] = inverse_order
    else:
        res = None

    return {
        "request": {
            "m1": m1,
            "m2": m2,
            "corpus": corpus,
        },
        "result": res,
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
