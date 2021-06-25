"""Automatically create all datasets for the deployed app using default path configurations"""

from typing import *
import typer
import argparse
import multiprocessing as mp
from pathlib import Path
from script_helpers.preprocess import preprocess
from script_helpers.compare_models_on_dataset import compare_models_on_dataset
from script_helpers.create_modelXdataset import create_analysis_results

from tqdm import tqdm
from itertools import combinations
import path_fixes as pf

parser = argparse.ArgumentParser()
parser.add_argument("--nworkers", "-w", type=int, default=4, help="Number of processes to run at once. E.g., 'This many models+datasets that can fit in memory'")
parser.add_argument("--topk", "-k", type=int, default=10, help="Top k results to record as top predictions")
parser.add_argument("--max_clamp_rank", "-r", type=int, default=50, help="Where to cutoff the clamped rank difference")
args = parser.parse_args()

TOPK = args.topk
MAX_CLAMP_RANK = args.max_clamp_rank

# `(model, type, dataset)` where `dataset` is the name of the `.txt` file in pf.DATASETS
CONFIG = [
    ('gpt2', 'gpt', 'mrpc_1+2.txt'),
    ('gpt2', 'gpt', 'commonsense-qa.txt'),
    ('gpt2', 'gpt', 'wino-bias.txt'),
    ('gpt2', 'gpt', 'wino-bias-with-addendum.txt'),
    ('gpt2', 'gpt', 'gpt2-gen-discrete.txt'),
    ('distilgpt2', 'gpt', 'mrpc_1+2.txt'),
    ('distilgpt2', 'gpt', 'commonsense-qa.txt'),
    ('distilgpt2', 'gpt', 'wino-bias.txt'),
    ('distilgpt2', 'gpt', 'wino-bias-with-addendum.txt'),
    ('distilgpt2', 'gpt', 'gpt2-gen-discrete.txt'),
    ('lysandre/arxiv', 'gpt', 'mrpc_1+2.txt'),
    ('lysandre/arxiv', 'gpt', 'commonsense-qa.txt'),
    ('lysandre/arxiv-nlp', 'gpt', 'mrpc_1+2.txt'),
    ('lysandre/arxiv-nlp', 'gpt', 'commonsense-qa.txt'),
    ('bert-base-cased', 'bert-cased', 'mrpc_1+2.txt'),
    ('bert-base-cased', 'bert-cased', 'commonsense-qa.txt'),
    ('bert-base-cased', 'bert-cased', 'wino-bias.txt'),
    ('bert-base-cased', 'bert-cased', 'wino-bias-with-addendum.txt'),
    ('bert-base-cased', 'bert-cased', 'short-jokes-small.txt'),
    ('distilbert-base-cased', 'bert-cased', 'mrpc_1+2.txt'),
    ('distilbert-base-cased', 'bert-cased', 'commonsense-qa.txt'),
    ('distilbert-base-cased', 'bert-cased', 'wino-bias.txt'),
    ('distilbert-base-cased', 'bert-cased', 'wino-bias-with-addendum.txt'),
    ('distilbert-base-cased', 'bert-cased', 'short-jokes-small.txt'),
    ('bert-base-uncased', 'bert-uncased', 'mrpc_1+2.txt'),
    ('bert-base-uncased', 'bert-uncased', 'commonsense-qa.txt'),
    ('bert-base-uncased', 'bert-uncased', 'wino-bias.txt'),
    ('bert-base-uncased', 'bert-uncased', 'wino-bias-with-addendum.txt'),
    ('distilbert-base-uncased', 'bert-uncased', 'mrpc_1+2.txt'),
    ('distilbert-base-uncased', 'bert-uncased', 'commonsense-qa.txt'),
    ('distilbert-base-uncased', 'bert-uncased', 'wino-bias.txt'),
    ('distilbert-base-uncased', 'bert-uncased', 'wino-bias-with-addendum.txt'),
    ('distilbert-base-uncased-finetuned-sst-2-english', 'bert-uncased', 'mrpc_1+2.txt'),
    ('distilbert-base-uncased-finetuned-sst-2-english', 'bert-uncased', 'commonsense-qa.txt'),
    ('distilbert-base-uncased-finetuned-sst-2-english', 'bert-uncased', 'wino-bias.txt'),
    ('distilbert-base-uncased-finetuned-sst-2-english', 'bert-uncased', 'wino-bias-with-addendum.txt'),
    # ('lysandre/arxiv-nlp', 'gpt', 'gpt2-gen-continuous.txt'),
    # ('lysandre/arxiv-nlp', 'gpt', 'gpt2-gen-discrete.txt'),
    # ('distilgpt2', 'gpt', 'short-jokes.txt'),
    # ('distilgpt2', 'gpt', 'gpt2-gen-continuous.txt'),
    # ('lysandre/arxiv', 'gpt', 'gpt2-gen-continuous.txt'),
    # ('lysandre/arxiv', 'gpt', 'gpt2-gen-discrete.txt'),
    # ('distilbert-base-uncased', 'bert-uncased', 'short-jokes.txt'),
    # ('bert-base-uncased', 'bert-uncased', 'short-jokes.txt'),
    # ('distilbert-base-uncased-finetuned-sst-2-english', 'bert-uncased', 'short-jokes.txt'),
    # ('gpt2', 'gpt', 'short-jokes.txt'),
    # ('gpt2', 'gpt', 'gpt2-gen-continuous.txt'),
]

# ==== CREATE HDF5 FILES ====
bad_combos = []
arg_pairs = [(m, ds) for m, _, ds in CONFIG]
def pooled_analysis(arg_pair):
    m, ds = arg_pair
    typer.echo(f"Starting `({m}, {ds})`")
    
    fargs = {
        "model_name": m,
        "dataset_path": pf.DATASETS / ds,
        "top_k": TOPK,
        # "max_clamp_rank": args.max_clamp_rank,
    }
    try:
        outfname = create_analysis_results(**fargs)
    except FileExistsError as e:
        typer.echo(f"File for ({m}, {ds}) already exists. Skipping.")
        outfname = e.details['outfname']
    except Exception as e:
        typer.echo(f"Unexpected issue for ({m}, {ds}) below:\n---\n{e}\n---\n")
        bad_combos.append(f"({m}, {ds})")
        outfname = None

    return outfname

with mp.Pool(args.nworkers) as p:
    outfnames = p.map(pooled_analysis, arg_pairs)

# ==== COMPARE MODELS ====
config_with_fnames = [c + (o,) for c, o in zip(CONFIG, outfnames)]

def make_comparison_args(conf: List[Tuple[str, str, str, Union[str, Path, None]]]):
    """
    Convert config (model, type, dataset_name, dataset_path) into (ds1, ds2) arguments that can be directly passed to the comparison scripts.

    Merge all compatible types and datasets.
    """
    # Temporary data structure
    config_dict = {}
    for model, typ, dataset, ds_path in conf:
        dataset_collection = config_dict.get(dataset, {})
        path_list = dataset_collection.get(typ, [])
        if ds_path is not None:
            path_list.append(ds_path)
        dataset_collection[typ] = path_list
        config_dict[dataset] = dataset_collection

    # Make arg triplets
    arg_list = []
    for dataset, types in config_dict.items():
        for typ, paths in types.items():
            path_combos = list(combinations(paths, 2))
            for path_combo in path_combos:
                arg_list.append(
                    (path_combo[0], path_combo[1])
                )

    return arg_list

compare_arg_list = make_comparison_args(config_with_fnames)
print(compare_arg_list)

bad_comparisons = []
def pooled_comparison(compare_args):
    x = {
        "ds1": compare_args[0],
        "ds2": compare_args[1],
        "max_clamp_rank": args.max_clamp_rank
    }

    try:
        return compare_models_on_dataset(**x)
    except Exception as e:
        print(f"Uh oh on {x['ds1']} and {x['ds2']}.\n---\n{e}\n---\n")
        bad_comparisons.append(x)
        return None
        

# with mp.Pool(args.nworkers) as p:
#     compare_outfnames = p.map(pooled_comparison, compare_arg_list)

compare_outfnames = [pooled_comparison(a) for a in compare_arg_list]

typer.echo("DONE")
bad_combo_strs = '\n'.join(bad_combos)
typer.echo(f"Ran into issues for the following analyses:\n---\n {bad_combo_strs}")

typer.echo(f"Ran into issues for the following comparisons:\n---\n {bad_comparisons}")