import pickle
import numpy as np

np.random.seed(1)

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

xval_train, xval_test = kfold[0]

# Dataset training
inds_u_train = inds_u[xval_train]
inds_i_train = inds_i[xval_train]
y_train = y[xval_train]
n_train = len(y_train)

# Dataset testing
inds_u_test = inds_u[xval_test]
inds_i_test = inds_i[xval_test]
y_test = y[xval_test]

# RMSE MF: 0.9004 	MAE: 0.7091
# RMSE ANN: 0.9438 	MAE: 0.7469

# Create config for model
hybrid_config = HybridConfig(
    n_factors=40,
    reg_bias_mf=0.00005,
    reg_latent=0.00002,
    reg_bias_cs=0.0001,
    reg_att_bias=0.0015,
    implicit_thresh_init=0.4,
    implicit_thresh_xtrain=0.7,
    opt_mf_init='nadam',
    opt_cs_init='nadam',
    opt_mf_xtrain='adadelta',
    opt_cs_xtrain='adadelta',
    batch_size_init_mf=512,
    batch_size_init_cs=512,
    batch_size_xtrain_mf=256,
    batch_size_xtrain_cs=1024,
    val_split_init=0.05,
    val_split_xtrain=0.2,
    xtrain_fsize_mf=0.2,
    xtrain_fsize_cs=0.2
)

test_while_fit = True

# Create model
model = HybridModel(users_features, items_features, hybrid_config, verbose=0)

if test_while_fit:
    model.fit([inds_u_train, inds_i_train], y_train, [inds_u_test, inds_i_test], y_test)
else:
    model.fit([inds_u_train, inds_i_train], y_train)

    rmse_mf, rmse_cs = model.test([inds_u_test, inds_i_test], y_test, True)
