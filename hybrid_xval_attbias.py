import pickle
import numpy as np

# np.random.seed(6)

# Local imports
import hybrid_model_attbias
from hybrid_model_attbias import HybridModel
import util

hybrid_model_attbias.verbose = 0

(inds_u, inds_i, y, users_features, items_features) = pickle.load(open('data/ml100k.pickle', 'rb'))

n_users, n_users_features = users_features.shape
n_items, n_items_features = items_features.shape

# items_features_norm = items_features / np.maximum(1, np.sum(items_features, axis=1)[:, None])

# Rescale ratings to ~(0.0, 1.0)
y = (y - 0.5) * 0.2

# Crossvalidation
n_fold = 5
user_coldstart = True
if user_coldstart:
    kfold = util.kfold_entries(n_fold, inds_u)
    # kfold = util.kfold_entries_plus(n_fold, inds_u, 3)
else:
    kfold = util.kfold(n_fold, inds_u)

rmses_mf_init = []
rmses_cs_init = []

rmses_mf_xtrain = []
rmses_cs_xtrain = []

for xval_train, xval_test in kfold:

    # Create model
    model = HybridModel(users_features, items_features)

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
    rmse_mf, rmse_cs = model.test(inds_u_test, inds_i_test, y_test, True)

    rmses_mf_init.append(rmse_mf)
    rmses_cs_init.append(rmse_cs)

    if user_coldstart:
        history_mf = model.step_mf(1.5)

    model.xtraining_complete()

    rmse_mf, rmse_cs = model.test(inds_u_test, inds_i_test, y_test, True)

    rmses_mf_xtrain.append(rmse_mf)
    rmses_cs_xtrain.append(rmse_cs)

    print('')

print('Crossval RMSE of MF (before xtrain): {:.4f} ±{:.4f}'.format(np.mean(rmses_mf_init), np.std(rmses_mf_init)))
print('Crossval RMSE of CS (before xtrain): {:.4f} ±{:.4f}'.format(np.mean(rmses_cs_init), np.std(rmses_cs_init)))
print('Crossval RMSE of MF: {:.4f} ±{:.4f}'.format(np.mean(rmses_mf_xtrain), np.std(rmses_mf_xtrain)))
print('Crossval RMSE of CS: {:.4f} ±{:.4f}'.format(np.mean(rmses_cs_xtrain), np.std(rmses_cs_xtrain)))
