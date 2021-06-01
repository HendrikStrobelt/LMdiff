"""
Example Usage: 

python scripts/create_modelXdataset.py distilgpt2 data/datasets/glue_mrpc_1+2.txt -o data/analysis_results --force_overwrite
"""
import argparse
from pathlib import Path
from typing import *
from tqdm import tqdm
import h5py
from analysis.analysis_pipeline import AutoLMPipeline, collect_analysis_info
from analysis.analysis_results_dataset import H5AnalysisResultDataset
from analysis.text_dataset import TextDataset
import path_fixes as pf


parser = argparse.ArgumentParser()
parser.add_argument(
    "model_name",
    type=str,
    help="Name (or path) of HF pretrained model to be loaded with AutoModelForCausalLM",
)
parser.add_argument("dataset_path", type=str, help="Path to dataset file")
parser.add_argument(
    "--output_d",
    "-o",
    type=str,
    default=str(pf.ANALYSIS),
    help="Which directory to store the output h5 file (default is the required config for this project). The name of the file is created using the name of provided dataset and model",
)
parser.add_argument(
    "--force_overwrite",
    action="store_true",
    help="If provided, overwrite the output file",
)
parser.add_argument(
    "--first_n",
    type=int,
    default=None,
    help="For testing. Only parse this many lines from provided dataset",
)
parser.add_argument(
    "--top_k",
    type=int,
    default=10,
    help="In place of each token calculate the topk logits/probs from the model",
)

args = parser.parse_args()


def create_analysis_results(
    outfname: Union[Path, str],
    dataset_fname,
    model_name,
    force_overwrite: bool,
    topk: int,
    first_n: Optional[int] = None,
):
    """
    
    Args:
        first_n: If provided, only evaluate this many sentences from the dataset
    """
    outfname = Path(outfname)
    tds = TextDataset.load(dataset_fname)
    pipeline = AutoLMPipeline.from_pretrained(model_name)
    v_sorted = {
        v: k
        for k, v in sorted(pipeline.tokenizer.vocab.items(), key=lambda item: item[1])
    }
    vocab = list(v_sorted.values())

    if outfname.exists() and not force_overwrite:
        raise ValueError(
            f"Will not overwrite existing '{outfname}' unless 'force_overwrite' is True"
        )
    if force_overwrite and outfname.exists():
        outfname.unlink()

    h5f = h5py.File(str(outfname), "w")

    # Create the attrs
    h5f.attrs["dataset_name"] = tds.name
    h5f.attrs["dataset_checksum"] = tds.checksum
    h5f.attrs["vocab_hash"] = str(pipeline.vocab_hash)
    h5f.attrs["model_name"] = model_name

    # Add vocabulary
    h5f.create_dataset("vocabulary", data=vocab)

    # Add content
    content = tds.content if first_n is None else tds.content[:first_n]
    for i, ex in tqdm(enumerate(content), total=len(content)):
        grp = h5f.create_group(H5AnalysisResultDataset.tokey(i))
        out = pipeline.forward(ex)
        out = collect_analysis_info(out, k=topk)
        out.save_to_h5group(grp)

    return H5AnalysisResultDataset(h5f)


# Smart defaults
dataset_path = Path(args.dataset_path)

if args.first_n is None:
    default_name = f"{dataset_path.stem}{pf.ANALYSIS_DELIM}{args.model_name}.h5"
else:
    default_name = f"{dataset_path.stem}{pf.ANALYSIS_DELIM}{args.model_name}_first{args.first_n}.h5"

output_d = pf.ANALYSIS if args.output_d is None else Path(args.output_d)
output_d.mkdir(parents=True, exist_ok=True)
output_f = output_d / default_name
output_f.parent.mkdir(parents=True, exist_ok=True)

create_analysis_results(
    output_f,
    dataset_path,
    args.model_name,
    topk=args.top_k,
    first_n=args.first_n,
    force_overwrite=args.force_overwrite,
)

