import regex as re
from typing import Union
from pathlib import Path
import spacy

nlp = spacy.load("en_core_web_sm")


def corpus2chunks(corpus_fname: Union[str, Path], n: int):
    """Convert a corpus file into chunks of size n
    
    Example:
        corpus_file = "./wizard-of-oz.txt"
        chunks = corpus2chunks(corpus_file, 3)
    """
    with open(corpus_fname) as f:
        out = f.read()
        print("Starting spacy processing of document")
        doc = nlp(out)
        print("Finished spacy processing.")
        sentences = [s.text for s in doc.sents]

    chunked_sents = [
        remove_newlines_and_spaces(" ".join(b)) for b in batch_list(sentences, n)
    ]
    return chunked_sents


def batch_list(input, n):
    out = []
    in_len = len(input)
    i = n
    while True:
        out.append(input[i - n : i])
        i += n
        if i >= in_len:
            break

    return out


def remove_newlines_and_spaces(s):
    out = re.sub("\n+", " ", s)
    out = re.sub(" +", " ", out)
    return out
