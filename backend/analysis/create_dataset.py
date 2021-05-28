from pathlib import Path
from typing import *
import datasets
from analysis import TextDataset

def create_text_dataset(ds:datasets.Dataset, name:str, ds2str:Callable[[datasets.Dataset], str], outfpath:Optional[str]=None):
    """
    Create a simple text dataset from a huggingface datasets. 
    
    NOTE: Dataset contents must fit into memory
    
    Args:
        ds: The HF dataset instance to convert
        name: What to name the dataset
        ds2str: A function that converts the dataset into a block of text with newlines separating examples
        outfpath: Where to save the dataset. If a directory (not given), save with the name in the location specified (current directory)

    Usage:
        ```
        import datasets
        ds = datasets.load_dataset("hate_offensive", split="train")
        def hate_ds_tweets(ds):
            # Choose to strip URLs, mentions, phrases that are too short...
            return "\n".join(ds['tweet'])
        
        create_text_dataset(ds, "hate-tweets", ds2str=hate_ds_tweets, outfpath="../data/datasets/")  
        ```
    """
    content = ds2str(ds)
    tds = TextDataset(content, {"name": name, "type": "human_created"})

    if outfpath is None:
        outfpath = name + ".txt"
    outfpath = Path(outfpath)
    if outfpath.is_dir():
        outfpath = outfpath / f"{name}.txt"
        
    outfpath.parent.mkdir(parents=True, exist_ok=True)
    tds.save(outfpath)
    return tds