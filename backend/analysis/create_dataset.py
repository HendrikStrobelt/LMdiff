from pathlib import Path
from typing import *
import datasets
from analysis import TextDataset

def content2saved_dataset(content: str, name: str, ds_type: str, outfpath:str=None):
    """Save a TextDataset with content, name, and type
    
    Args:
        content: Newline-separated string contents of the dataset
        name: Name of the dataset
        ds_type: Either "human_created" or "machine_generated""
        outfpath: Where to save the file. If a directory, create a file with name `{name}.txt`. If a file, save there.

    Returns:
        TextDataset with corresponding content and specified headers
    """
    tds = TextDataset(content, {"name": name, "type": ds_type})

    if outfpath is None:
        outfpath = name + ".txt"
    outfpath = Path(outfpath)
    if outfpath.is_dir():
        outfpath = outfpath / f"{name}.txt"
        
    outfpath.parent.mkdir(parents=True, exist_ok=True)
    tds.save(outfpath)
    return tds

def create_text_dataset_from_object(obj: List[str], name: str, ds_type="human_created", outfpath:str=None):
    """
    Create a simple text dataset from a list of strings to treat as content.
    
    NOTE: Dataset contents must fit into memory
    
    Args:
        fname: Name of the file containing text sentences to analyze
        name: What to name the dataset
        ds_type: Either "human_created" or "machine_generated""
        outfpath: Where to save the dataset. If a directory (not given), save with the name in the location specified (current directory)
    """
    content = "\n".join(obj)
    tds = content2saved_dataset(content, name, ds_type=ds_type, outfpath=outfpath)
    return tds


def create_text_dataset_from_file(fname, name: str, ds_type="human_created", outfpath:str=None):
    """
    Create a simple text dataset from a text file with a new phrase on each line
    
    NOTE: Dataset contents must fit into memory
    
    Args:
        fname: Name of the file containing text sentences to analyze
        name: What to name the dataset
        ds_type: One of "human_created" or "machine_generated". Default to "human_created"
        outfpath: Where to save the dataset. If a directory (not given), save with the name in the location specified (current directory)
    """
    with open(fname, 'r') as fp:
        content = fp.read()
    tds = content2saved_dataset(content, name, ds_type=ds_type, outfpath=outfpath)
    return tds


def create_text_dataset_from_hf_datasets(ds:datasets.Dataset, name:str, ds2str:Callable[[datasets.Dataset], str], ds_type="human_created", outfpath:Optional[str]=None):
    """
    Create a simple text dataset from a huggingface datasets. 
    
    NOTE: Dataset contents must fit into memory
    
    Args:
        ds: The HF dataset instance to convert
        name: What to name the dataset
        ds2str: A function that converts the dataset into a block of text with newlines separating examples
        ds_type: One of "human_created" or "machine_generated". Default to "human_created"
        outfpath: Where to save the dataset. If a directory (not given), save with the name in the location specified (current directory)

    Usage:
        ```
        import datasets
        ds = datasets.load_dataset("hate_offensive", split="train")
        def hate_ds_tweets(ds):
            # Choose to strip URLs, mentions, phrases that are too short...
            return "\n".join(ds['tweet'])
        
        create_text_dataset_from_hf_datasets(ds, "hate-tweets", ds2str=hate_ds_tweets, outfpath="../data/datasets/")  
        ```
    """
    content = ds2str(ds)
    tds = content2saved_dataset(content, name, ds_type=ds_type, outfpath=outfpath)
    return tds