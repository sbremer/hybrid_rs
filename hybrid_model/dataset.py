import pickle

_datasets = {'ml100k': 'data/ml100k', 'ml1m': 'data/ml1m'}


def _get_dataset_filename(dataset, desc=False):

    if dataset not in _datasets.keys():
        raise ValueError

    if desc:
        filename = _datasets[dataset] + '_desc.pickle'
    else:
        filename = _datasets[dataset] + '.pickle'

    return filename


class Dataset:
    def __init__(self, inds_u, inds_i, y, users_features, items_features):
        self.data = (inds_u, inds_i, y, users_features, items_features)
        self.n_users, self.n_users_features = users_features.shape
        self.n_items, self.n_items_features = items_features.shape


def get_dataset(dataset):

    filename = _get_dataset_filename(dataset)

    (inds_u, inds_i, y, users_features, items_features) = pickle.load(open(filename, 'rb'))

    return Dataset(inds_u, inds_i, y, users_features, items_features)


def get_dataset_desc(dataset):

    filename = _get_dataset_filename(dataset, True)

    (users_desc, items_desc, users_features_desc, items_features_desc) = pickle.load(open(filename, 'rb'))

    return (users_desc, items_desc, users_features_desc, items_features_desc)
