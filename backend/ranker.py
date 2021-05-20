import numpy as np

def sort_by_keys(xs, keys, reverse=True):
    """Sort list of xs by a list of identical length keys
    
    Args:
        xs: List of items to rank
        keys: Numerical values against which to sort.
        reverse: If True, sort by descending order (we want the examples of highestt difference). Otherwise, sort ascending (default True)

    Returns:
        A sorted list of xs
    """
    assert len(xs) == len(keys), "Expect every x to have a corresponding key"
    return [x for _, x in sorted(zip(keys, xs), reverse=reverse)]

def get_kl(x): 
    kl = x['prob']['kl']
    return np.array(kl).mean()

def get_av_diff(x):
    """Get absolute value of the probability differences"""
    diff = np.array(x['prob']['diff'])
    return np.mean(np.abs(diff))

# Map from strings to function handles that take one input argument and returns a number
METHOD_MAP = {
    'kl': get_kl,
    'avdiff': get_av_diff,
}