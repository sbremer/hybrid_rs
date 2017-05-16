import numpy as np


def full(x_test, y_test, user_dist, item_dist):
    return x_test, y_test


# Hacky summing up
stats_user = [[] for _ in range(10)]
stats_item = [[] for _ in range(10)]


def binning_user(n_bins, b):
    if b >= n_bins:
        raise ValueError('b has to be within range(n_bins)')

    def bin_user(x_test, y_test, user_dist, item_dist):
        users_sorted = np.argsort(-user_dist)
        user_bin = np.array_split(users_sorted, n_bins)[b]
        select = np.in1d(x_test[0], user_bin)

        # print('Avg #ratings/user for user bin {}/{}: {}'.format(b+1, n_bins, np.mean(user_dist[user_bin])))
        stats_user[b].append(np.mean(user_dist[user_bin]))

        return [x_test[0][select], x_test[1][select]], y_test[select]

    return bin_user


def binning_item(n_bins, b):
    if b >= n_bins:
        raise ValueError('b has to be within range(n_bins)')

    def bin_item(x_test, y_test, user_dist, item_dist):
        items_sorted = np.argsort(-item_dist)
        item_bin = np.array_split(items_sorted, n_bins)[b]
        select = np.in1d(x_test[1], item_bin)

        # print('Avg #ratings/item for item bin {}/{}: {}'.format(b + 1, n_bins, np.mean(item_dist[item_bin])))
        stats_item[b].append(np.mean(item_dist[item_bin]))

        return [x_test[0][select], x_test[1][select]], y_test[select]

    return bin_item
