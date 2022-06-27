'''
purpose_: Helper function to get indices on data split - Copied from StackOverflow
author_: StackOverflow
'''

import random
import numpy as np

# Get split indices
def split(split_: list, len_: int, seed: int):
    split_ = [round(each_split, 2) for each_split in split_]
    if sum(split_) != 1.00:
        raise Exception("split_ do not sum to one!")

    # to shuffle ordered list
    indices = list(range(len_))
    random.Random(seed).shuffle(indices)
    indices = np.array(indices, dtype=int)

    # get groups
    n_split_ = []
    for i in range(len(split_) - 1):
        n_split_.append(int(max(split_[i] * len_, 0)))
    n_split_.append(int(max(len_ - sum(n_split_), 0)))

    if sum(n_split_) != len_:
        raise Exception("n_split_ do not sum to len_!")

    # sample indices
    n_selected = 0
    indices_split_ = []
    for n_frac in n_split_:
        indices_frac = indices[n_selected:n_selected + n_frac]
        indices_split_.append(indices_frac)
        n_selected += n_frac

    # Check no intersections
    for a, indices_frac_A in enumerate(indices_split_):
        for b, indices_frac_B in enumerate(indices_split_):
            if a == b:
                continue
            if is_intersect(indices_frac_A, indices_frac_B):
                raise Exception("there are intersections!")

    return indices_split_

# Is there intersection?
def is_intersect(arr1, arr2):
    n_intersect = len(np.intersect1d(arr1, arr2))
    if n_intersect == 0: return False
    else: return True