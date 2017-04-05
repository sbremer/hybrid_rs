import pickle
import numpy as np

# Local imports
import hybrid_model
from hybrid_model import HybridModel
import util

hybrid_model.verbose = 2
hybrid_model.batch_size = 1024
hybrid_model.val_split = 0.25
hybrid_model.bias_mf = True
hybrid_model.bias_ann = True


# (meta_users, meta_items, U, I, Y) = pickle.load(open('data/ratings_metadata.pickle', 'wb'))
(meta_users, meta_items) = pickle.load(open('data/imdb_metadata.pickle', 'rb'))
(_, inds_u, inds_i, y) = pickle.load(open('data/cont.pickle', 'rb'))

# meta_items = meta_items[:, 19:21]

# Normalize features and set Nans to zero (=mean)
meta_users = (meta_users - np.nanmean(meta_users, axis=0)) / np.nanstd(meta_users, axis=0)
meta_items = (meta_items - np.nanmean(meta_items, axis=0)) / np.nanstd(meta_items, axis=0)
meta_users[np.isnan(meta_users)] = 0
meta_items[np.isnan(meta_items)] = 0

# Rescale ratings to ~(0.0, 1.0)
y_org = y.copy()
y = (y - 0.5) * 0.2

# Crossvalidation
n_fold = 4
user_coldstart = True
if user_coldstart:
    kfold = util.kfold_entries(n_fold, inds_u)
else:
    kfold = util.kfold(n_fold, inds_u)

xval_train, xval_test = next(kfold)

# Create model
model = HybridModel(meta_users, meta_items)

# Dataset training
inds_u_train = inds_u[xval_train]
inds_i_train = inds_i[xval_train]
y_train = y[xval_train]
n_train = len(y_train)

# Dataset testing
inds_u_test = inds_u[xval_test]
inds_i_test = inds_i[xval_test]
y_test = y[xval_test]

# Run initial (separate) training
model.train_initial(inds_u_train, inds_i_train, y_train, True)
print('Testing using test set before cross-training:')
rmse_mf, rmse_ann = model.test(inds_u_test, inds_i_test, y_test, True)

# Cross-train with half as many matrix entries than actual training set samples
n_xtrain = int(n_train * 1.0)

# Alternating cross-training
for i in range(20):
    # ANN step
    model.step_ann(int(n_xtrain), True)
    # MF step
    model.step_mf(int(n_xtrain), True)

    # Test
    print('Results after training step {}:'.format(i + 1))
    rmse_mf, rmse_ann = model.test(inds_u_test, inds_i_test, y_test, True)
