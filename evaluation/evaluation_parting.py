from typing import List, Tuple
import numpy as np


class Parting:
    def part(self, x_test: List[np.ndarray], y_test: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        raise NotImplementedError

    def update(self, *args):
        pass


class Full(Parting):
    def part(self, x_test, y_test):
        return x_test, y_test


class Binning(Parting):
    def __init__(self, n_bins, b):
        if b >= n_bins:
            raise ValueError('b has to be within range(n_bins)')

        self.n_bins = n_bins
        self.b = b
        self.user_dist = None
        self.item_dist = None

    def part(self, x_test: List[np.ndarray], y_test: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        raise NotImplementedError

    def update(self, user_dist, item_dist):
        self.user_dist = user_dist
        self.item_dist = item_dist


# Hacky summing up
stats_user = [[] for _ in range(10)]
stats_item = [[] for _ in range(10)]


class BinningUser(Binning):
    def part(self, x_test, y_test):
        users_sorted = np.argsort(-self.user_dist)
        user_bin = np.array_split(users_sorted, self.n_bins)[self.b]
        select = np.in1d(x_test[0], user_bin)

        # print('Avg #ratings/user for user bin {}/{}: {}'.format(b+1, n_bins, np.mean(user_dist[user_bin])))
        stats_user[self.b].append(np.mean(self.user_dist[user_bin]))

        return [x_test[0][select], x_test[1][select]], y_test[select]


class BinningItem(Binning):
    def part(self, x_test, y_test):
        items_sorted = np.argsort(-self.item_dist)
        item_bin = np.array_split(items_sorted, self.n_bins)[self.b]
        select = np.in1d(x_test[1], item_bin)

        # print('Avg #ratings/item for item bin {}/{}: {}'.format(b + 1, n_bins, np.mean(item_dist[item_bin])))
        stats_item[self.b].append(np.mean(self.item_dist[item_bin]))

        return [x_test[0][select], x_test[1][select]], y_test[select]
