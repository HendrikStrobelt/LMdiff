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


def strip_doublelines(content):
    return re.sub(r"\n+", "\n", content)


def hash_content(content: str):
    return sha256(content.encode("utf-8")).hexdigest()


class TextDataset:
    valid_types = set(["machine_generated", "human_created"])

    def __init__(self, content, metadata):
        self.content = re.sub(r"\n+", r"\n", content).split("\n")
        self.metadata = metadata
        self.type = metadata.get("type", None)
        if self.type not in self.valid_types:
            raise ValueError(
                f"Unknown type of dataset '{self.type}'. Expected one of {self.valid_types}"
            )
        self.name = metadata.get("name", None)
        self.checksum = metadata.get("checksum", None)

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

        post = frontmatter.Post(self.content, metadata)
        with open(fname, "w") as fp:
            frontmatter.dump(post, fp)

    @property
    def frontmatter(self) -> Dict[str, any]:
        return self.metadata