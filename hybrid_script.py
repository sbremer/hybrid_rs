import pickle
import numpy as np

# Local imports
from hybrid_model import HybridModel
import util


# (meta_users, meta_items, U, I, Y) = pickle.load(open('data/ratings_metadata.pickle', 'wb'))
(meta_users, meta_items) = pickle.load(open('data/imdb_metadata.pickle', 'rb'))
(_, inds_u, inds_i, y) = pickle.load(open('data/cont.pickle', 'rb'))

# Normalize features and set Nans to zero (=mean)
meta_users = (meta_users - np.nanmean(meta_users, axis=0)) / np.nanstd(meta_users, axis=0)
meta_items = (meta_items - np.nanmean(meta_items, axis=0)) / np.nanstd(meta_items, axis=0)
meta_users[np.isnan(meta_users)] = 0
meta_items[np.isnan(meta_items)] = 0

# Rescale ratings to ~(0.0, 1.0)
y_org = y.copy()
y = (y - 0.5) * 0.2

# Create model
model = HybridModel(meta_users, meta_items)

# Run initial (separate) training
model.train_initial(inds_u, inds_i, y, True)

n_train = len(y)

# Cross-train with half as many matrix entries than actual training set samples
n_xtrain = int(n_train / 2)

# Alternating cross-training
for i in range(5):
    print('Training step {}'.format(i + 1))
    # MF step
    model.step_mf(n_xtrain)

    # ANN step
    model.step_ann(n_xtrain)

    # Test
    print('Results after training step {}:'.format(i + 1))
    rmse_mf, rmse_ann = model.test(inds_u, inds_i, y, True)
