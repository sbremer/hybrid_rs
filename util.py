import numpy as np
from sklearn.model_selection import KFold


def kfold(k, entries):
    kf = KFold(n_splits=k, shuffle=True)
    return kf.split(entries)


def kfold_entries(k, entries):
    unique = np.unique(entries)
    kf = KFold(n_splits=k, shuffle=True)
    for train_indices_entries, test_indices_entries in kf.split(unique):
        train_indices = [i for i, x in enumerate(entries) if x in train_indices_entries]
        test_indices = [i for i, x in enumerate(entries) if x in test_indices_entries]
        yield train_indices, test_indices
