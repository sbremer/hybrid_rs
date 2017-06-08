import pickle

import numpy as np

np.random.seed(0)

# Local imports
from hybrid_model.hybrid import HybridModel, HybridConfig
from hybrid_model import transform
from hybrid_model.index_sampler import IndexSamplerUserbased
from evaluation.evaluation import EvaluationResults
import util

(inds_u, inds_i, y, users_features, items_features) = pickle.load(open('data/ml100k.pickle', 'rb'))

n_users, n_users_features = users_features.shape
n_items, n_items_features = items_features.shape

# items_features_norm = items_features / np.maximum(1, np.sum(items_features, axis=1)[:, None])

# Crossvalidation
n_fold = 5
user_coldstart = True
if user_coldstart:
    kfold = util.kfold_entries(n_fold, inds_u)
    # kfold = util.kfold_entries_plus(n_fold, inds_u, 3)
else:
    kfold = util.kfold(n_fold, inds_u)

kfold = list(kfold)

from sklearn.model_selection import ParameterGrid

param_grid_setup = dict(n_factors=[40],
    reg_bias_mf=[0.00005],
    reg_latent=[0.00003],
    reg_bias_cs=[0.0001],
    reg_att_bias=[0.0015],
    implicit_thresh_init=[0.7],
    implicit_thresh_xtrain=[0.85],
    opt_mf_init=['nadam'],
    opt_cs_init=['nadam'],
    opt_mf_xtrain=['adadelta'],
    opt_cs_xtrain=['adadelta'],
    batch_size_init_mf=[512],
    batch_size_init_cs=[1024],
    batch_size_xtrain_mf=[256],
    batch_size_xtrain_cs=[1024],
    val_split_init=[0.05],
    val_split_xtrain=[0.05],
    index_sampler=[IndexSamplerUserbased],
    xtrain_patience=[5],
    xtrain_max_epochs=[10],
    xtrain_data_shuffle=[False],
    transformation=[transform.TransformationLinear]
)

param_grid = list(ParameterGrid(param_grid_setup))
n_grid = len(param_grid)

print('Testing {} combinations for configuration'.format(n_grid))

rmses_mf_grid = np.zeros((n_grid,))
rmses_cs_grid = np.zeros((n_grid,))

for i, config in enumerate(param_grid):
    hybrid_config = HybridConfig(**config)

    print('Now testing config {}: {}'.format(i, hybrid_config))

    results_before_xtrain = EvaluationResults()
    results_after_xtrain = EvaluationResults()

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

        model.fit_init([inds_u_train, inds_i_train], y_train)
        result_before_xtrain = model.evaluate([inds_u_test, inds_i_test], y_test)

        model.fit_cross([inds_u_train, inds_i_train], y_train)
        result_after_xtrain = model.evaluate([inds_u_test, inds_i_test], y_test)

        print(result_after_xtrain)
        results_before_xtrain.add(result_before_xtrain)
        results_after_xtrain.add(result_after_xtrain)

    rmses_mf_grid[i] = results_after_xtrain.mean_rmse_mf()
    rmses_cs_grid[i] = results_after_xtrain.mean_rmse_cs()

    print('Config before_xtrain {}: '.format(i), results_before_xtrain)
    print('Config after_xtrain {}: '.format(i), results_after_xtrain)

i_mf_best = np.argmin(rmses_mf_grid)
i_cs_best = np.argmin(rmses_cs_grid)

print('Best performance for MF {}: {}'.format(i_mf_best, rmses_mf_grid[i_mf_best]))
print(param_grid[i_mf_best])
print('Best performance for CS {}: {}'.format(i_cs_best, rmses_cs_grid[i_cs_best]))
print(param_grid[i_cs_best])
