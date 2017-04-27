import numpy as np


class IndexSampler:
    """
    Generates indices for user-item pairs which are not in the training set.
    Build for sparse training set! Will be very inefficient else!
    """

    def __init__(self, user_dist, item_dist, x_train):
        self.n_users = len(user_dist)
        self.n_items = len(item_dist)

        # Create and fill entry lookup table
        self.lookup = {}
        for u_i in zip(x_train[0], x_train[1]):
            self.lookup[u_i] = True

        # Calculate sample probability

        self.prob_from_mf_user = user_dist / np.sum(user_dist)
        self.prob_from_mf_item = item_dist / np.sum(item_dist)

        user_counts_adj = np.maximum(4, user_dist) ** 2
        # item_counts_adj = np.maximum(4, item_dist) ** 2

        item_counts_adj = np.ones((self.n_items,))

        self.prob_from_ann_user = (1 / user_counts_adj) / np.sum(1 / user_counts_adj)
        self.prob_from_ann_item = (1 / item_counts_adj) / np.sum(1 / item_counts_adj)

    def get_indices(self, n_indices):
        inds_u = np.zeros((n_indices,), np.int)
        inds_i = np.zeros((n_indices,), np.int)

        lookup_samples = {}

        got = 0
        while got < n_indices:
            u = np.random.randint(self.n_users)
            i = np.random.randint(self.n_items)

            if (u, i) not in self.lookup and (u, i) not in lookup_samples:
                inds_u[got] = u
                inds_i[got] = i
                lookup_samples[(u, i)] = True
                got += 1

        return inds_u, inds_i

    def get_indices_from_mf(self, n_indices):
        inds_u = np.zeros((n_indices,), np.int)
        inds_i = np.zeros((n_indices,), np.int)

        lookup_samples = {}

        got = 0
        while got < n_indices:
            u = np.random.choice(np.arange(self.n_users), p=self.prob_from_mf_user)
            i = np.random.choice(np.arange(self.n_items), p=self.prob_from_mf_item)

            if (u, i) not in self.lookup and (u, i) not in lookup_samples:
                inds_u[got] = u
                inds_i[got] = i
                lookup_samples[(u, i)] = True
                got += 1

        return inds_u, inds_i

    def get_indices_from_cs(self, n_indices):
        inds_u = np.zeros((n_indices,), np.int)
        inds_i = np.zeros((n_indices,), np.int)

        lookup_samples = {}

        got = 0
        while got < n_indices:
            u = np.random.choice(np.arange(self.n_users), p=self.prob_from_ann_user)
            i = np.random.choice(np.arange(self.n_items), p=self.prob_from_ann_item)

            if (u, i) not in self.lookup and (u, i) not in lookup_samples:
                inds_u[got] = u
                inds_i[got] = i
                lookup_samples[(u, i)] = True
                got += 1

        return inds_u, inds_i