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


class IndexGen:
    """
    Generates indices for user-item pairs which are not in the training set.
    Build for sparse training set! Might be very inefficient else.
    """

    def __init__(self, n_users, n_items, U, I):
        assert len(U) == len(I)

        self.n_users = n_users
        self.n_items = n_items

        # Create and fill entry lookup table
        self.lookup = {}
        for u_i in zip(U, I):
            self.lookup[u_i] = True

    def get_indices(self, n_indices):
        U = np.zeros((n_indices,))
        I = np.zeros((n_indices,))

        got = 0
        while got < n_indices:
            u = np.random.randint(self.n_users)
            i = np.random.randint(self.n_items)
            if (u, i) not in self.lookup:
                U[got] = u
                I[got] = i
                got += 1

        return U, I
