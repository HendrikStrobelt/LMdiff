from typing import *
import h5py
from analysis.analysis_pipeline import LMAnalysisOutputH5
import numpy as np
from pathlib import Path

class H5AnalysisResultDataset:
    @staticmethod
    def tokey(i: int):
        """Convert an integer to index into the correct group"""
        return str(i).zfill(15)
        
    def __init__(self, h5f:h5py.File):
        self.h5f = h5f
        self.dataset_name = self.h5f.attrs['dataset_name']
        self.dataset_checksum = self.h5f.attrs['dataset_checksum']
        self.model_name = self.h5f.attrs['model_name']
        self.vocab_hash = self.h5f.attrs['vocab_hash']

        self.__n = 0 # For iteration

        # Create the vocabulary
        self.vocab = self.h5f["vocabulary"]
        
    @classmethod
    def from_file(cls, fname: Union[Path, str]):
        h5f = h5py.File(fname, 'r')
        return cls(h5f)

    def is_comparable(self, other):
        return (self.dataset_checksum == other.dataset_checksum) and (self.vocab_hash == other.vocab_hash)

    def __del__(self):
        try:
            self.h5f.close()
        except TypeError as e:
            pass
        
    def _grp2output(self, grp):
        return LMAnalysisOutputH5.from_group(grp)
    
    def __len__(self):
        return len(self.h5f.keys()) - 1 # for vocabulary

    def __getitem__(self, val:Union[int, slice, list, np.ndarray]) -> Union[LMAnalysisOutputH5, List[LMAnalysisOutputH5]]:
        if isinstance(val, int):
            grp = self.h5f[self.tokey(val)]
            return self._grp2output(grp)
        elif isinstance(val, slice):
            idxs = np.arange(len(self))[val]
            grps = (self.h5f[self.tokey(idx)] for idx in idxs)
            return [self._grp2output(g) for g in grps]
        elif isinstance(val, list):
            grps = (self.h5f[self.tokey(int(idx))] for idx in val)
            return [self._grp2output(g) for g in grps]
        elif isinstance(val, np.ndarray):
            if not np.issubdtype(val.dtype, np.integer):
                raise ValueError(f"Needs an integer-like index. Got {val.dtype}")
            grps = (self.h5f[self.tokey(int(idx))] for idx in val)
            return [self._grp2output(g) for g in grps]
        else:
            raise ValueError(f"Indexed with {type(val)}. Not a slice or int or list or np.array of integers!")

    def __iter__(self):
        self.__n = 0
        return self

    def __next__(self):
        if self.__n < len(self):
            result = self[self.__n]
            self.__n += 1
            return result
        else:
            raise StopIteration