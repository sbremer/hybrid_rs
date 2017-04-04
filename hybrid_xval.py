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

# Crossvalidation
n_fold = 5
user_coldstart = False
if user_coldstart:
    kfold = util.kfold_entries(n_fold, inds_u)
else:
    kfold = util.kfold(n_fold, inds_u)

rmses_mf = []
rmses_ann = []

for xval_train, xval_test in kfold:
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

    # Cross-train with half as many matrix entries than actual training set samples
    n_xtrain = int(n_train / 2)

    rmse_mf, rmse_ann = float('nan'), float('nan')

    # Alternating cross-training
    for i in range(5):
        print('Training step {}'.format(i + 1))
        # MF step
        model.step_mf(n_xtrain)

        # ANN step
        model.step_ann(n_xtrain)

        # Test
        print('Results after training step {}:'.format(i + 1))
        rmse_mf, rmse_ann = model.test(inds_u_test, inds_i_test, y_test, True)

    rmses_mf.append(rmse_mf)
    rmses_ann.append(rmse_ann)


print('Crossval RMSE of MF: {0:.4f}'.format(np.mean(rmses_mf)))
print('Crossval RMSE of ANN: {0:.4f}'.format(np.mean(rmses_ann)))
