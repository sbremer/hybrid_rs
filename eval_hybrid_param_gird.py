import pickle
import numpy as np

np.random.seed(0)

# Local imports
from hybrid_model.hybrid import HybridModel, HybridConfig
import util

(inds_u, inds_i, y, users_features, items_features) = pickle.load(open('data/ml100k.pickle', 'rb'))

n_users, n_users_features = users_features.shape
n_items, n_items_features = items_features.shape

# items_features_norm = items_features / np.maximum(1, np.sum(items_features, axis=1)[:, None])

# Rescale ratings to ~(0.0, 1.0)
y = (y - 0.5) * 0.2

# Crossvalidation
n_fold = 5
user_coldstart = False
if user_coldstart:
    kfold = util.kfold_entries(n_fold, inds_u)
    # kfold = util.kfold_entries_plus(n_fold, inds_u, 3)
else:
    kfold = util.kfold(n_fold, inds_u)

kfold = list(kfold)

from sklearn.model_selection import ParameterGrid

param_grid_setup = dict(n_factors=[50],
    reg_bias_mf=[0.00002, 0.00005, 0.0001],
    reg_latent=[0.00005],
    reg_bias_cs=[0.00002, 0.00005, 0.0001],
    reg_att_bias=[0.002],
    implicit_thresh_init=[0.4],
    implicit_thresh_xtrain=[0.7],
    opt_mf_init=['adadelta'],
    opt_cs_init=['nadam'],
    opt_mf_xtrain=['adadelta'],
    opt_cs_xtrain=['adadelta'],
    batch_size_init=[512],
    batch_size_xtrain_mf=[512],
    batch_size_xtrain_cs=[512],
    val_split_init=[0.05],
    val_split_xtrain=[0.2],
    xtrain_fsize_mf=[0.2],
    xtrain_fsize_cs=[0.2]
)

param_grid = list(ParameterGrid(param_grid_setup))
n_grid = len(param_grid)

print('Testing {} combinations for configuration'.format(n_grid))

rmses_mf_grid = np.zeros((n_grid,))
rmses_cs_grid = np.zeros((n_grid,))

for i, config in enumerate(param_grid):
    hybrid_config = HybridConfig(**config)

    print('Now testing config {}: {}'.format(i, hybrid_config))

    # Init xval
    rmses_mf = []
    rmses_cs = []

    for xval_train, xval_test in kfold:
        # Dataset training
        inds_u_train = inds_u[xval_train]
        inds_i_train = inds_i[xval_train]
        y_train = y[xval_train]
        n_train = len(y_train)

        # Dataset testing
        inds_u_test = inds_u[xval_test]
        inds_i_test = inds_i[xval_test]
        y_test = y[xval_test]

        # Create model
        model = HybridModel(users_features, items_features, hybrid_config, verbose=0)

        model.fit([inds_u_train, inds_i_train], y_train)

        rmse_mf, rmse_cs = model.test([inds_u_test, inds_i_test], y_test, True)

        rmses_mf.append(rmse_mf)
        rmses_cs.append(rmse_cs)

    rmse_mf = np.mean(rmses_mf)
    rmse_cs = np.mean(rmses_cs)
    rmses_mf_grid[i] = rmse_mf
    rmses_cs_grid[i] = rmse_cs

    print('Config {}: MF {:.4f}  CS {:.4f}'.format(i, rmse_mf, rmse_cs))

i_mf_best = np.argmin(rmses_mf_grid)
i_cs_best = np.argmin(rmses_cs_grid)

print('Best performance for MF {}: {}'.format(i_mf_best, rmses_mf_grid[i_mf_best]))
print(param_grid[i_mf_best])
print('Best performance for CS {}: {}'.format(i_cs_best, rmses_cs_grid[i_cs_best]))
print(param_grid[i_cs_best])
