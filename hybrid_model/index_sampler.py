import numpy as np


class IndexSampler:
    def __init__(self, user_dist, item_dist, x_train):
        self.n_users = len(user_dist)
        self.n_items = len(item_dist)

        # Create and fill entry lookup table
        self.lookup = {}
        for u_i in zip(x_train[0].flatten(), x_train[1].flatten()):
            self.lookup[u_i] = True

    def get_indices_from_cf(self):
        raise NotImplementedError

    def get_indices_from_md(self):
        raise NotImplementedError


class IndexSamplerUserbased(IndexSampler):
    def __init__(self, user_dist, item_dist, x_train):
        super().__init__(user_dist, item_dist, x_train)

        min_ratings = 40

        # 15: 0.8932
        # 20: 0.8925
        # 25: 0.8925
        # 30: 0.8923
        # 35: 0.8926
        # 40: 0.8923
        # 50: 0.8932

        sample_users = user_dist <= min_ratings
        self.users_cs = np.arange(self.n_users)[sample_users]
        self.user_dist_cs = np.maximum(0, min_ratings - user_dist[sample_users])*1

        self.item_dist = item_dist

        # Calculate sample probability
        self.prob_from_cf_user = user_dist / np.sum(user_dist)
        self.prob_from_cf_item = item_dist / np.sum(item_dist)

        # self.prob_from_md_item = np.ones((self.n_items,)) / self.n_items
        self.prob_from_md_item = item_dist / np.sum(item_dist)

        self.n_inds_from_cf = int(len(x_train[0]) * 0.15)
        self.n_inds_from_md = np.sum(self.user_dist_cs)
        print('n_inds_from_md = {}'.format(self.n_inds_from_md))

    def get_indices_from_cf(self):
        inds_u = np.zeros((self.n_inds_from_cf,1), np.int)
        inds_i = np.zeros((self.n_inds_from_cf,1), np.int)

        lookup_samples = {}

        got = 0
        while got < self.n_inds_from_cf:
            u = np.random.choice(np.arange(self.n_users), p=self.prob_from_cf_user)
            i = np.random.choice(np.arange(self.n_items), p=self.prob_from_cf_item)

            if (u, i) not in self.lookup and (u, i) not in lookup_samples:
                inds_u[got] = u
                inds_i[got] = i
                lookup_samples[(u, i)] = True
                got += 1

        return inds_u, inds_i

    def get_indices_from_md(self):
        inds_u = np.zeros((self.n_inds_from_md,1), np.int)
        inds_i = np.zeros((self.n_inds_from_md,1), np.int)

        lookup_samples = {}

        got = 0

        for u, n_u in zip(self.users_cs, self.user_dist_cs):
            got_u = 0
            while got_u < n_u:
                i = np.random.choice(np.arange(self.n_items), p=self.prob_from_md_item)
                if (u, i) not in self.lookup and (u, i) not in lookup_samples:
                    inds_u[got] = u
                    inds_i[got] = i
                    lookup_samples[(u, i)] = True
                    got += 1
                    got_u += 1

        return inds_u, inds_i


class IndexSamplerUniform(IndexSampler):

    def __init__(self, user_dist, item_dist, x_train, n_inds_from_cf, n_inds_from_md):
        # Make sure this is called through a child class
        if self.__class__ == IndexSamplerUniform:
            raise NotImplementedError

        super().__init__(user_dist, item_dist, x_train)

        self.n_inds_from_cf = n_inds_from_cf
        self.n_inds_from_md = n_inds_from_md

        # To be overwritten by subclass
        self.prob_from_cf_user = None
        self.prob_from_cf_item = None
        self.prob_from_md_user = None
        self.prob_from_md_item = None

    def get_indices_from_cf(self):
        inds_u = np.zeros((self.n_inds_from_cf,), np.int)
        inds_i = np.zeros((self.n_inds_from_cf,), np.int)

        lookup_samples = {}

        got = 0
        while got < self.n_inds_from_cf:
            u = np.random.choice(np.arange(self.n_users), p=self.prob_from_cf_user)
            i = np.random.choice(np.arange(self.n_items), p=self.prob_from_cf_item)

            if (u, i) not in self.lookup and (u, i) not in lookup_samples:
                inds_u[got] = u
                inds_i[got] = i
                lookup_samples[(u, i)] = True
                got += 1

        return inds_u, inds_i

    def get_indices_from_md(self):
        inds_u = np.zeros((self.n_inds_from_md,), np.int)
        inds_i = np.zeros((self.n_inds_from_md,), np.int)

        lookup_samples = {}

        got = 0
        while got < self.n_inds_from_md:
            u = np.random.choice(np.arange(self.n_users), p=self.prob_from_md_user)
            i = np.random.choice(np.arange(self.n_items), p=self.prob_from_md_item)

            if (u, i) not in self.lookup and (u, i) not in lookup_samples:
                inds_u[got] = u
                inds_i[got] = i
                lookup_samples[(u, i)] = True
                got += 1
        return inds_u, inds_i


class IndexSampler1(IndexSamplerUniform):
    def __init__(self, user_dist, item_dist, x_train, n_inds_from_cf, n_inds_from_md):
        super().__init__(user_dist, item_dist, x_train, n_inds_from_cf, n_inds_from_md)

        # Calculate sample probability
        self.prob_from_cf_user = user_dist / np.sum(user_dist)
        self.prob_from_cf_item = item_dist / np.sum(item_dist)

        user_counts_adj = np.maximum(4, user_dist) ** 2
        item_counts_adj = np.maximum(4, item_dist) ** 2

        self.prob_from_md_user = (1 / user_counts_adj) / np.sum(1 / user_counts_adj)
        self.prob_from_md_item = (1 / item_counts_adj) / np.sum(1 / item_counts_adj)


class IndexSampler2(IndexSamplerUniform):
    def __init__(self, user_dist, item_dist, x_train):
        super().__init__(user_dist, item_dist, x_train, -1, -1)

        # Calculate sample probability
        user_dist_adj_from_md = np.maximum(1, 25 - user_dist)
        item_dist_adj_from_md = np.maximum(1, 30 - item_dist)
        self.prob_from_md_user = user_dist_adj_from_md / np.sum(user_dist_adj_from_md)
        self.prob_from_md_item = item_dist_adj_from_md / np.sum(item_dist_adj_from_md)

        self.prob_from_cf_user = user_dist / np.sum(user_dist)
        self.prob_from_cf_item = item_dist / np.sum(item_dist)

        self.n_inds_from_cf = int(len(x_train[0]) * 0.15)
        self.n_inds_from_md = int(np.sum(user_dist_adj_from_md) * 3)
