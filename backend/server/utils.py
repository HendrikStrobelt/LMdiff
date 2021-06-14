"""General programming utils, inclined toward functional programming.

If ever a function changes its input in place, it is denoted by a trailing `_`
"""

import inspect
import numpy as np
import torch
from itertools import zip_longest
from typing import List, Set, Union, Dict
from enum import Enum


class SortOrder(str, Enum):
    ascending = "ascending"
    descending = "descending"


def ifnone(*xs):
    """Return the first item in 'x' that is not None"""
    for x in xs:
        if x is not None:
            return x
    return None


def custom_dir(c, add):
    return dir(type(c)) + list(c.__dict__.keys()) + add


class GetAttr:
    """Base class for attr accesses in `self._xtra` passed down to `self.default`

    Taken from article by Jeremy Howard: https://www.fast.ai/2019/08/06/delegation/

    Usage:

        ```
        class ProductPage(GetAttr):
            def __init__(self, page, price, cost):
                self.page,self.price,self.cost = page,price,cost
                self.default = page
        ```
    """

    @property
    def _xtra(self):
        return [o for o in dir(self.default) if not o.startswith("_")]

    def __getattr__(self, k):
        if k in self._xtra:
            return getattr(self.default, k)
        raise AttributeError(k)

    def __dir__(self):
        return custom_dir(self, self._xtra)


# Can i delegate many different functions?
# Can i add a new docstring to the existing docstring of the delgated function? Or at least point to the function delegated?
def delegates(to=None, keep=False):
    """ Decorator: replace `**kwargs` in signature with params from `to`.

    Taken from article by Jeremy Howard: https://www.fast.ai/2019/08/06/delegation/
    """

    def _f(f):
        if to is None:
            to_f, from_f = f.__base__.__init__, f.__init__
        else:
            to_f, from_f = to, f
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)
        k = sigd.pop("kwargs")
        s2 = {
            k: v
            for k, v in inspect.signature(to_f).parameters.items()
            if v.default != inspect.Parameter.empty and k not in sigd
        }
        sigd.update(s2)
        if keep:
            sigd["kwargs"] = k
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f

    return _f


def pick(keys: Union[List, Set], obj: Dict) -> Dict:
    """ Return a NEW object containing `keys` from the original `obj` """
    return {k: obj[k] for k in keys}


def memoize(f):
    """Memoize a function.

    Use lookup table when the same inputs are passed to the function instead of running that function again
    """
    memo = {}

    def helper(*x):
        if x not in memo:
            memo[x] = f(*x)
        return memo[x]

    return helper


def assoc(k, v, orig):
    """Given an original dictionary orig, return a cloned dictionary with `k` set to `v`"""
    out = orig.copy()
    out[k] = v
    return out


def make_unique(f):
    """The input function will only run and return if it hasn't seen its argument before.

    Otherwise, it will return `None`.
    """
    s = set()

    def helper(x):
        if x in s:
            return None
        s.add(x)
        return f(x)

    return helper


def flatten_(items, seqtypes=(list, tuple)):
    """Flattten an arbitrarily nested list IN PLACE"""
    for i, x in enumerate(items):
        while i < len(items) and isinstance(items[i], seqtypes):
            items[i: i + 1] = items[i]
    return items


def map_nlist(f, nlist):
    """Map a function across an arbitrarily nested list"""
    new_list = []
    for i in range(len(nlist)):
        if isinstance(nlist[i], list):
            new_list += [map_nlist(f, nlist[i])]
        else:
            new_list += [f(nlist[i])]
    return new_list


def jsonify_np(obj):
    """Convert numpy object of any kind into a jsonable list"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def round_nested_list(x, ndigits: int = 3, force_float=True):
    """Round a nested list to `ndigits` decimals, coercing ints into floats"""
    if isinstance(x, float):
        return round(x, ndigits)
    elif isinstance(x, int):
        if force_float:
            return float(x)
        return x
    return [round_nested_list(i, ndigits, force_float) for i in x]


def deepdict_to_json(x, ndigits=3, force_float=False):
    """Convert a nested dictionary to a jsonable object, rounding items as necessary"""
    # First convert out of numpy
    x = jsonify_np(x)
    if isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        return round_nested_list(x.tolist(), ndigits, force_float)
    elif isinstance(x, list) or isinstance(x, tuple):
        return [deepdict_to_json(i, ndigits, force_float) for i in x]
    elif isinstance(x, float):
        return round(x, ndigits)
    elif isinstance(x, dict):
        return {k: deepdict_to_json(v, ndigits, force_float) for k, v in x.items()}

    return x
