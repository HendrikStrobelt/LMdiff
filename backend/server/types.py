from pydantic import BaseModel
import numpy as np
from typing import *

class HashableBaseModel(BaseModel):
    def __hash__(self):
        return hash(self.json())

    @classmethod
    def validate(cls, v: np.ndarray):
        return v

class GoodbyePayload(HashableBaseModel):
    firstname:str

class SuggestionsRequest(HashableBaseModel):
    m1: str
    m2: str
    corpus: str

class AnalyzeRequest(HashableBaseModel):
    m1: str
    m2: Optional[str]=None
    text: str