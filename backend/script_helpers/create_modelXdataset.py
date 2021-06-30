from typing import *
import path_fixes as pf
from pathlib import Path
from tqdm import tqdm
import h5py
from analysis import model_name2path, model_path2name
from analysis.analysis_pipeline import AutoLMPipeline, collect_analysis_info
from analysis.analysis_cache import AnalysisCache
from analysis.text_dataset import TextDataset

def analyze_dataset(
    outfname: Union[Path, str],
    dataset_fname:Union[Path, str],
    model_name:Union[Path,str],
    force_overwrite: bool,
    topk: int,
    first_n: Optional[int] = None,
    output_attentions=False):
    """Analyze a dataset with a huggingface model

    Raises:
        FileExistsError: Will not overwrite a file that already exists without `force_overwrite` set to True

    Returns:
        AnalysisCache that can be saved or used
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
        error = FileExistsError(
            f"Will not overwrite existing '{outfname}' unless 'force_overwrite' is True"
        )
        error.details = {}
        error.details['outfname'] = outfname
        raise error
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
        grp = h5f.create_group(AnalysisCache.tokey(i))
        out = pipeline.forward(ex, output_attentions=output_attentions)
        out = collect_analysis_info(out, k=topk)
        out.save_to_h5group(grp)

    return AnalysisCache(h5f)

def create_analysis_results(
    model_name: str,
    dataset_path: str,
    output_d: str = str(pf.ANALYSIS),
    force_overwrite: bool = False,
    first_n: Union[None, int] = None,
    top_k: int = 10,
):
    """Create analysis results of a transformers model on a dataset

    Args:
        model_name (str): Name (or path) of HF pretrained model to be loaded with AutoModelWithLMHead
        dataset_path (str): Path to dataset file
        output_d (str, optional): Which directory to store the output h5 file. The name of the file is created using the name of provided dataset and model. Default is the required config for this project.
        force_overwrite (bool, optional): If provided, overwrite the output HDF5 file (if exists). Defaults to False.
        first_n (Union[None, int], optional): If provided, only evaluate this many sentences from the dataset. For testing. Defaults to None.
        top_k (int, optional): [description]. In place of each token calculate the topk logits/probs from the model. Defaults to 10.
    """
    dataset_path = Path(dataset_path)

    model_save_name = model_name2path(model_name)

    if first_n is None:
        default_name = f"{dataset_path.stem}{pf.ANALYSIS_DELIM}{model_save_name}.h5"
    else:
        default_name = f"{dataset_path.stem}{pf.ANALYSIS_DELIM}{model_save_name}_first{first_n}.h5"

    output_d = pf.ANALYSIS if output_d is None else Path(output_d)
    output_d.mkdir(parents=True, exist_ok=True)
    output_f = output_d / default_name
    output_f.parent.mkdir(parents=True, exist_ok=True)

    analyze_dataset(
        output_f,
        dataset_path,
        model_name,
        topk=top_k,
        first_n=first_n,
        force_overwrite=force_overwrite,
    )
    
    return output_f