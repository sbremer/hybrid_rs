import pickle


class Dataset:
    def __init__(self, inds_u, inds_i, y, users_features, items_features):
        self.data = (inds_u, inds_i, y, users_features, items_features)
        self.n_users, self.n_users_features = users_features.shape
        self.n_items, self.n_items_features = items_features.shape


def get_dataset(dataset):
    datasets = {'ml100k': 'data/ml100k.pickle', 'ml1m': 'data/ml1m.pickle'}

    if dataset not in datasets.keys():
        raise ValueError

    (inds_u, inds_i, y, users_features, items_features) = pickle.load(open(datasets[dataset], 'rb'))

    return Dataset(inds_u, inds_i, y, users_features, items_features)
