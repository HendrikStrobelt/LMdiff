import numpy as np
from typing import *

def topk_token_diff(t1:List[List[int]], t2:List[List[int]]):
    topk_token_set1 = [set(t) for t in t1]
    topk_token_set2 = [set(t) for t in t2]
    n_topk_diff = np.array([len(s1.difference(s2)) for s1, s2 in zip(topk_token_set1, topk_token_set2)])
    return n_topk_diff