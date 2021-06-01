from pydantic import BaseModel
from typing import *

class SuggestionsRequest(BaseModel):
    m1: str
    m2: str
    corpus: str

class AnalyzeRequest(BaseModel):
    m1: str
    m2: Optional[str]=None
    text: str

class SpecificAttentionRequest(BaseModel):
    m1: str
    m2: str
    text: str
    token_index_in_text: int
    outward_attentions: bool = True # If true, return the attentions out of that token. If false, return attention toward that token from the other tokens