"""
We can pass a collection of examples through a language model to find instances
that qualitatively show the performance of that model on the collection of
examples.

This file defines the expected protocol for models interacting with a text
dataset
"""
from typing import *
import frontmatter
import regex as re
from hashlib import sha256
from dataclasses import dataclass


def strip_doublelines(content):
    return re.sub(r"\n+", "\n", content)


def hash_content(content: str):
    return sha256(content.encode("utf-8")).hexdigest()

class TextDataset:
    valid_types = set(["machine_generated", "human_created"])

    def __init__(self, content:str, metadata:dict):
        """

        Args:
            content (str): Data to store as the content of a dataset. New lines indicate new phrases
            metadata (dict): Contains, at a minimum, the keys "name" and "type"
        """        
        self.content = re.sub(r"\n+", r"\n", content).split("\n")
        self.metadata = metadata
        self.type = self.metadata.get('type', "human_created")

    def _check_type(self, new_type):
        if new_type not in self.valid_types:
            raise ValueError(
                f"Unknown type of dataset '{new_type}'. Expected one of {self.valid_types}"
            )

    @property
    def checksum(self) -> str:
        return self.metadata.get("checksum", None)

    @property
    def name(self) -> str:
        return self.metadata.get("name", None)

    @name.setter
    def name(self, val:str):
        self.metadata['name'] = val

    @property
    def type(self) -> str:
        return self.metadata.get("type", None)
    
    @type.setter
    def type(self, val: str):
        self._check_type(val)
        self.metadata['type'] = val


    @classmethod
    def load(cls, fname: str):
        with open(fname, "r", encoding="utf-8") as fp:
            fm = frontmatter.load(fp)

        return cls(fm.content, fm.metadata)

    def save(self, fname: str):
        new_checksum = hash_content("\n".join(self.content))

        metadata = {}
        metadata.update(self.metadata)
        metadata.update(
            {"name": self.name, "type": self.type, "checksum": new_checksum}
        )

        post = frontmatter.Post("\n".join(self.content), **metadata)

        with open(fname, "wb") as fp:
            frontmatter.dump(post, fp)

    def __len__(self):
        return len(self.content)

    @property
    def frontmatter(self) -> Dict[str, any]:
        return self.metadata

    def __repr__(self):
        return f"TextDataset(name={self.name}, type={self.type}, ...)"