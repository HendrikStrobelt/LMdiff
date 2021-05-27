from typing import *
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
# from server.utils import jsonify_np # Circular import issues
# from torch.nn.functional import kl_div

import torch
import h5py
from dataclasses import dataclass

def get_group(f:Union[h5py.File, h5py.Group], gname:str):
    if gname in f.keys():
        return f[gname]
    return f.create_group(gname)
    
#!! Monkey patching
h5py.File.get_group = get_group
h5py.Group.get_group = get_group

class AnalysisLMPipelineForwardOutput(CausalLMOutputWithCrossAttentions):
    def __init__(self, phrase:str, in_ids:torch.tensor, **kwargs):
        super().__init__(**kwargs)
        self.N = len(in_ids)
        self.in_ids = in_ids
        self.phrase = phrase

class AutoLMPipeline():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config = self.model.config
        self.vocab_hash = hash(frozenset(self.tokenizer.vocab.items()))
        self.device = self.model.device
                
    @classmethod
    def from_pretrained(cls, name_or_path):
        """Create a model and tokenizer from a single name, passing arguemnts to `from_pretrained` of the AutoTokenizer and AutoModel"""
        model = AutoModelForCausalLM.from_pretrained(name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        return cls(model, tokenizer)

    def for_model(self, s):
        iids = self.tokenizer.prepare_for_model(self.tokenizer.encode(s), return_tensors="pt")['input_ids']
        return iids
    
    def for_model_batch(self, s:List[str]):
        ii = self.tokenizer.batch_encode_plus(s)['input_ids']
        input = self.tokenizer.prepare_for_model(ii, return_tensors="pt", padding=True)

        return input
    
    def forward(self, s) -> AnalysisLMPipelineForwardOutput:
        with torch.no_grad():
            tids = self.for_model(s)
            
            output = self.model(tids, output_attentions=True)
            new_output = AnalysisLMPipelineForwardOutput(s, tids, **output.__dict__)
        
        return new_output
    
    def idmat2tokens(self, idmat:torch.tensor) -> List[str]:
        """ Convert arbitrarily nested IDs into tokens """
        output = []
        for i, idlist in enumerate(idmat):
            if idlist.ndim == 1:
                output.append(self.tokenizer.convert_ids_to_tokens(idlist))
        return output

@dataclass
class LMAnalysisOutputH5:
    token_ids: np.array # N
    ranks: np.array # N,
    probs: np.array # N,
    topk_prob_values: np.array # k, N
    topk_token_ids: np.array # k, N
    attention: np.array # Layer, Head, N, N
    phrase: str
    # Add attributes

    @classmethod
    def from_group(cls, grp):
        return cls(
            token_ids=np.array(grp['token_ids']),
            ranks=np.array(grp['ranks']),
            probs=np.array(grp['probs']),
            topk_prob_values=np.array(grp['topk_probs']),
            topk_token_ids=np.array(grp['topk_token_ids']),
            attention=np.array(grp['attention']),
            phrase=grp.attrs['phrase']
        )
    
    def save_to_h5group(self, h5group:h5py.Group):
        h5group.create_dataset("token_ids", data=self.token_ids)
        h5group.create_dataset("ranks", data=self.ranks)
        h5group.create_dataset("probs", data=self.probs)
        h5group.create_dataset("topk_probs", data=self.topk_prob_values)
        h5group.create_dataset("topk_token_ids", data=self.topk_token_ids)
        h5group.create_dataset("attention", data=self.attention)
        h5group.attrs['phrase'] = self.phrase

    # def to_json(self):
    #     return {
    #         "token_ids": jsonify_np(self.token_ids),
    #         "ranks": jsonify_np(self.ranks),
    #         "probs": jsonify_np(self.probs),
    #         "topk_probs": jsonify_np(self.topk_prob_values),
    #         "topk_token_ids": jsonify_np(self.topk_token_ids),
    #         "attention": jsonify_np(self.attention),
    #         "phrase": self.phrase
    #     }

    
@dataclass
class LMAnalysisOutput:
    token_ids: torch.tensor # N,
    ranks: torch.tensor # N,
    probs: torch.tensor # N,
    topk_prob_values: torch.tensor # k, N
    topk_token_ids: torch.tensor # k, N
    attention: torch.tensor # Layer, Head, N, N
    phrase: str
        
    def for_h5(self):
        return LMAnalysisOutputH5(
            token_ids=self.token_ids.cpu().numpy().astype(np.int64),
            ranks=self.ranks.cpu().numpy().astype(np.uint32),
            probs=self.probs.cpu().numpy().astype(np.float32),
            topk_prob_values=self.topk_prob_values.cpu().numpy().astype(np.float32),
            topk_token_ids=self.topk_token_ids.cpu().numpy().astype(np.int64),
            attention=self.attention.cpu().numpy().astype(np.float32),
            phrase=self.phrase,
        )
        
    def save_to_h5group(self, h5group:h5py.Group):
        self.for_h5().save_to_h5group(h5group)
        return h5group
    

def collect_analysis_info(model_output:AnalysisLMPipelineForwardOutput, k=10) -> LMAnalysisOutput:
    """
    Analyze the output of a language model for probabilities and ranks
    
    Args:
        model_output: The output of the causal Language Model
        k: The number of top probabilities we care about
    """
    probs = torch.softmax(model_output.logits, dim=1)
    ranks = (torch.argsort(model_output.logits, dim=1, descending=True) == model_output.in_ids.unsqueeze(1).expand_as(model_output.logits)).nonzero()[:,1]
    phrase_probs = probs[torch.arange(model_output.N), model_output.in_ids]
#     topk_logit_values, topk_logit_inds = torch.topk(output.logits, k=k, dim=1)
    topk_prob_values, topk_prob_inds = torch.topk(probs, k=k, dim=1)
    attention = torch.cat(model_output.attentions, dim=0) # Layer, Head, N, N
    
    return LMAnalysisOutput(
        phrase=model_output.phrase,
        token_ids=model_output.in_ids,
        ranks=ranks,
        probs=phrase_probs,
        topk_prob_values=topk_prob_values,
        topk_token_ids=topk_prob_inds,
        attention=attention,
    )

def analyze_text(text:str, pp1:AutoLMPipeline, pp2:AutoLMPipeline, topk=10):
    assert pp1.vocab_hash == pp2.vocab_hash, "Vocabularies of the two pipelines must align"

    tokens = pp1.tokenizer.tokenize(text)
    output1 = pp1.forward(text)
    output2 = pp2.forward(text)

    parsed_output1 = collect_analysis_info(output1, k=topk)
    parsed_output2 = collect_analysis_info(output2, k=topk)

    return {
        "text": text,
        "tokens": tokens,
        "m1": {
            "rank": parsed_output1.ranks,
            "prob": parsed_output1.probs,
            "topk": pp1.idmat2tokens(parsed_output1.topk_token_ids), # Turn to str
            "attentions": parsed_output1.attention
        },
        "m2": {
            "rank": parsed_output2.ranks,
            "prob": parsed_output2.probs,
            "topk": pp2.idmat2tokens(parsed_output2.topk_token_ids), # Turn to str
            "attentions": parsed_output2.attention
        },
        "diff": {
            "rank": parsed_output2.ranks - parsed_output1.ranks,
            "prob": parsed_output2.probs - parsed_output1.probs,
            # "kl": kl_div(parsed_output1.probs, parsed_output2.probs, reduction="sum") # No meaning
        }
    }