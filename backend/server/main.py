import argparse
import os
import json
import re
from pathlib import Path
from typing import *
from functools import lru_cache
import datasets

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import server.api as api
from server.utils import deepdict_to_json
from analysis import AutoLMPipeline, analyze_text
import path_fixes as pf

from api import LMComparer, ModelManager

__author__ = "DreamTeam V1.5: Hendrik Strobelt, Sebastian Gehrmann, Ben Hoover"


@lru_cache
def get_pipeline(name: str):
    return AutoLMPipeline.from_pretrained(name)


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

# Main routes
@app.get("/")
def index():
    """For local development, serve the index.html in the dist folder"""
    return RedirectResponse(url="client/index.html")


# the `file_path:path` says to accept any path as a string here. Otherwise, `file_paths` containing `/` will not be served properly
@app.get("/client/{file_path:path}")
def send_static_client(file_path: str):
    """ Serves (makes accessible) all files from ./client/ to ``/client/{path}``. Used primarily for development. NGINX handles production.

    Args:
        path: Name of file in the client directory
    """
    f = str(pf.DIST / file_path)
    print("Finding file: ", f)
    return FileResponse(f)


@app.get("/data/{path:path}")
def send_data(path):
    """ serves all files from the data dir to ``/data/{path:path}``

    Args:
        path: Path from api call
    """
    f = Path(get_args().dir) / path
    print("Finding data file: ", f)
    return FileResponse(f)


# ======================================================================
## MAIN API ##
# ======================================================================
@app.get("/api/available_datasets")
def get_available_datasets():
    return ["hate-tweets"]


@app.get("/api/all_projects")
def get_all_projects():
    res = [
        {"model": "gpt2"},
        # {"model": "lysandre/arxiv-nlp"},
        {"model": "distilgpt2"},
        # {"model": "lysandre/arxiv"},
    ]

    # for k in projects.keys():
    #     res[k] = projects[k].config
    return res


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

    return {"request": {"m1": m1, "m2": m2, "corpus": corpus,}, "result": res}


@app.post("/api/analyze-text")
def analyze_models_on_text(payload: api.AnalyzeRequest):
    m1 = payload.get("m1")
    m2 = payload.get("m2")
    pp1 = get_pipeline(m1)
    pp2 = get_pipeline(m2)
    text = payload.get("text")
    output = analyze_text(text, pp1, pp2)
    result = deepdict_to_json(output, ndigits=4)

    res = {"request": {"m1": m1, "m2": m2, "text": text}, "result": result}
    return res


@app.post("/api/analyze")
def analyze(payload: api.AnalyzeRequest):
    m1 = payload.get("m1")
    m2 = payload.get("m2")
    text = payload.get("text")

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
