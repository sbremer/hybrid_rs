import numpy as np
import itertools
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


def kfold_entries_plus(k, entries, plus):
    unique = np.unique(entries)
    kf = KFold(n_splits=k, shuffle=True)
    for train_indices_entries, test_indices_entries in kf.split(unique):

        train = {key: [] for key in train_indices_entries}
        test = {key: [] for key in test_indices_entries}
        for i, x in enumerate(entries):
            if x in train_indices_entries:
                train[x].append(i)
            if x in test_indices_entries:
                test[x].append(i)

        plus_list = []

        for test_key in test.keys():
            for _ in range(plus):
                n = len(test[test_key])
                if n <= 1:
                    break
                i = np.random.randint(n - 1)
                element = test[test_key].pop(i)
                plus_list.append(element)

        train_indices = np.array(list(itertools.chain(plus_list, *train.values())))
        test_indices = np.array(list(itertools.chain(*test.values())))

        yield train_indices, test_indices
