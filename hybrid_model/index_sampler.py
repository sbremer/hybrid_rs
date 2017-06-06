import numpy as np


class IndexSampler:
    def __init__(self, user_dist, item_dist, x_train):
        self.n_users = len(user_dist)
        self.n_items = len(item_dist)

        # Create and fill entry lookup table
        self.lookup = {}
        for u_i in zip(x_train[0], x_train[1]):
            self.lookup[u_i] = True

    def get_indices_from_cf(self):
        raise NotImplementedError

    def get_indices_from_md(self):
        raise NotImplementedError


class IndexSamplerUserbased(IndexSampler):
    def __init__(self, user_dist, item_dist, x_train):
        super().__init__(user_dist, item_dist, x_train)

        min_ratrings = 40

        # 15: 0.8932
        # 20: 0.8925
        # 25: 0.8925
        # 30: 0.8923
        # 35: 0.8926
        # 40: 0.8923
        # 50: 0.8932

        sample_users = user_dist <= min_ratrings
        self.users_cs = np.arange(self.n_users)[sample_users]
        self.user_dist_cs = np.maximum(0, min_ratrings - user_dist[sample_users])

        self.item_dist = item_dist

        # Calculate sample probability
        self.prob_from_mf_user = user_dist / np.sum(user_dist)
        self.prob_from_mf_item = item_dist / np.sum(item_dist)

        self.n_inds_from_mf = int(len(x_train[0]) * 0.15)
        self.n_inds_from_cs = np.sum(self.user_dist_cs)
        print('n_inds_from_cs = {}'.format(self.n_inds_from_cs))

    def get_indices_from_cf(self):
        inds_u = np.zeros((self.n_inds_from_mf,), np.int)
        inds_i = np.zeros((self.n_inds_from_mf,), np.int)

        lookup_samples = {}

        got = 0
        while got < self.n_inds_from_mf:
            u = np.random.choice(np.arange(self.n_users), p=self.prob_from_mf_user)
            i = np.random.choice(np.arange(self.n_items), p=self.prob_from_mf_item)

            if (u, i) not in self.lookup and (u, i) not in lookup_samples:
                inds_u[got] = u
                inds_i[got] = i
                lookup_samples[(u, i)] = True
                got += 1

        return inds_u, inds_i

    def get_indices_from_md(self):
        inds_u = np.zeros((self.n_inds_from_cs,), np.int)
        inds_i = np.zeros((self.n_inds_from_cs,), np.int)

        lookup_samples = {}

        got = 0

        for u, n_u in zip(self.users_cs, self.user_dist_cs):
            got_u = 0
            while got_u < n_u:
                i = np.random.choice(np.arange(self.n_items), p=self.prob_from_mf_item)
                if (u, i) not in self.lookup and (u, i) not in lookup_samples:
                    inds_u[got] = u
                    inds_i[got] = i
                    lookup_samples[(u, i)] = True
                    got += 1
                    got_u += 1

        return inds_u, inds_i


class IndexSamplerUniform(IndexSampler):

    def __init__(self, user_dist, item_dist, x_train, n_inds_from_mf, n_inds_from_cs):
        # Make sure this is called through a child class
        if self.__class__ == IndexSamplerUniform:
            raise NotImplementedError

        super().__init__(user_dist, item_dist, x_train)

        self.n_inds_from_mf = n_inds_from_mf
        self.n_inds_from_cs = n_inds_from_cs

    def get_indices_from_cf(self):
        inds_u = np.zeros((self.n_inds_from_mf,), np.int)
        inds_i = np.zeros((self.n_inds_from_mf,), np.int)

        lookup_samples = {}

        got = 0
        while got < self.n_inds_from_mf:
            u = np.random.choice(np.arange(self.n_users), p=self.prob_from_mf_user)
            i = np.random.choice(np.arange(self.n_items), p=self.prob_from_mf_item)

            if (u, i) not in self.lookup and (u, i) not in lookup_samples:
                inds_u[got] = u
                inds_i[got] = i
                lookup_samples[(u, i)] = True
                got += 1

        return inds_u, inds_i

    def get_indices_from_md(self):
        inds_u = np.zeros((self.n_inds_from_cs,), np.int)
        inds_i = np.zeros((self.n_inds_from_cs,), np.int)

        lookup_samples = {}

        got = 0
        while got < self.n_inds_from_cs:
            u = np.random.choice(np.arange(self.n_users), p=self.prob_from_cs_user)
            i = np.random.choice(np.arange(self.n_items), p=self.prob_from_cs_item)

            if (u, i) not in self.lookup and (u, i) not in lookup_samples:
                inds_u[got] = u
                inds_i[got] = i
                lookup_samples[(u, i)] = True
                got += 1
        return inds_u, inds_i


class IndexSampler1(IndexSamplerUniform):
    def __init__(self, user_dist, item_dist, x_train, n_inds_from_mf, n_inds_from_cs):
        super().__init__(user_dist, item_dist, x_train, n_inds_from_mf, n_inds_from_cs)

        # Calculate sample probability
        self.prob_from_mf_user = user_dist / np.sum(user_dist)
        self.prob_from_mf_item = item_dist / np.sum(item_dist)

        user_counts_adj = np.maximum(4, user_dist) ** 2
        item_counts_adj = np.maximum(4, item_dist) ** 2

        self.prob_from_cs_user = (1 / user_counts_adj) / np.sum(1 / user_counts_adj)
        self.prob_from_cs_item = (1 / item_counts_adj) / np.sum(1 / item_counts_adj)


class IndexSampler2(IndexSamplerUniform):
    def __init__(self, user_dist, item_dist, x_train):

        # Calculate sample probability
        user_dist_adj_from_cs = np.maximum(1, 25 - user_dist)
        item_dist_adj_from_cs = np.maximum(1, 30 - item_dist)
        self.prob_from_cs_user = user_dist_adj_from_cs / np.sum(user_dist_adj_from_cs)
        self.prob_from_cs_item = item_dist_adj_from_cs / np.sum(item_dist_adj_from_cs)

        self.prob_from_mf_user = user_dist / np.sum(user_dist)
        self.prob_from_mf_item = item_dist / np.sum(item_dist)

        n_inds_from_mf = int(len(x_train[0]) * 0.15)
        n_inds_from_cs = int(np.sum(user_dist_adj_from_cs) * 3)

        super().__init__(user_dist, item_dist, x_train, n_inds_from_mf, n_inds_from_cs)

