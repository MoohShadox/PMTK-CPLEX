"""
@author: Ouaguenouni Mohamed
"""
import numpy as np


def sample_subset(items, size=None):
    """
    This function samples a subset of the given set of items.
    By default the size of the subset is random but could be specified
    with the parameter size.
    Params:
        -items: The set of items from which the subset is sampled.
        -size: Size of each subset.
    """
    if not size:
        size = np.random.randint(1, len(items))
    subset = np.random.choice(items, size)
    subset = tuple(sorted(set(subset)))
    while(len(subset) != size and len(subset) <= len(items)):
        subset = np.random.choice(items, size)
        subset = tuple(sorted(set(subset)))
    return subset


def sample_subsets(items, size=None, n_subsets=1):
    """
    This function samples a sequence of subsets of the given set of items.
    By default the size of the subset is random but could be specified
    with the parameter size.
    Params:
        -items: The set of items from which the subset is sampled.
        -size: Size of each subset.
        -n_subsets: The number of subsets to sample.
    """
    subsets = []
    cpt = 0
    while len(subsets) < n_subsets and len(subsets) < 2**(len(items)):
        cpt = cpt + 1
        subset = sample_subset(items)
        if subset not in subsets:
            subsets.append(subset)
        if cpt > 100000:
            break
    return subsets


def sample_subset_values(items, n_subsets=1):
    """
    Sample a sequence of subsets and a continuous value for each.
    Params:
        -items: The set of items from which the subset is sampled.
        -n_subsets: The number of subsets to sample.
        -value_range: The range in which each subset value is.
    """
    subsets = sample_subsets(items, n_subsets=n_subsets)
    subset_values = {s: np.random.rand() for s in subsets}
    return subset_values
