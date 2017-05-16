import pickle
import numpy as np

np.random.seed(0)

# Local imports
from hybrid_model.hybrid import HybridModel, HybridConfig
from hybrid_model import transform
from hybrid_model.index_sampler import IndexSampler2, IndexSamplerUserbased
import util

(inds_u, inds_i, y, users_features, items_features) = pickle.load(open('data/ml100k.pickle', 'rb'))

n_users, n_users_features = users_features.shape
n_items, n_items_features = items_features.shape

# items_features_norm = items_features / np.maximum(1, np.sum(items_features, axis=1)[:, None])

# Crossvalidation
n_fold = 5
user_coldstart = False
if user_coldstart:
    kfold = util.kfold_entries(n_fold, inds_u)
    # kfold = util.kfold_entries_plus(n_fold, inds_u, 2)
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

# RMSE MF: 0.9011 	MAE: 0.7133
# RMSE ANN: 0.9436 	MAE: 0.7474

# Create config for model
hybrid_config = HybridConfig(
    n_factors=40,
    reg_bias_mf=0.00005,
    reg_latent=0.00004,#2
    reg_bias_cs=0.0001,
    reg_att_bias=0.0015,
    implicit_thresh_init=0.7,
    implicit_thresh_xtrain=0.8,
    opt_mf_init='nadam',
    opt_cs_init='nadam',
    opt_mf_xtrain='adadelta',
    opt_cs_xtrain='adadelta',
    batch_size_init_mf=512,
    batch_size_init_cs=1024,
    batch_size_xtrain_mf=256,
    batch_size_xtrain_cs=1024,
    val_split_init=0.05,
    val_split_xtrain=0.05,
    index_sampler=IndexSamplerUserbased,
    xtrain_patience=5,
    xtrain_max_epochs=10,
    xtrain_data_shuffle=False,
    transformation=transform.TransformationLinear
)

# None
# RMSE MF: 0.9097 	MAE: 0.7159 	NDCG: 0.5091
# RMSE ANN: 0.9343 	MAE: 0.7391 	NDCG: 0.4852
# Linear
# RMSE MF: 0.8940 	MAE: 0.7042 	NDCG: 0.5227
# RMSE ANN: 0.9336 	MAE: 0.7380 	NDCG: 0.4842
# LinearShift
# RMSE MF: 0.9540 	MAE: 0.7407 	NDCG: 0.4953
# RMSE ANN: 0.9684 	MAE: 0.7536 	NDCG: 0.4836
# Quad
# RMSE MF: 0.9087 	MAE: 0.7093 	NDCG: 0.5156
# RMSE ANN: 0.9442 	MAE: 0.7385 	NDCG: 0.4837

test_while_fit = False

# Create model
model = HybridModel(users_features, items_features, hybrid_config, verbose=0)

if test_while_fit:
    model.fit([inds_u_train, inds_i_train], y_train, [inds_u_test, inds_i_test], y_test)
else:
    model.fit_init_only([inds_u_train, inds_i_train], y_train)

    print('Before xtrain:')
    result = model.evaluate([inds_u_test, inds_i_test], y_test)
    print(result)

    model.fit_xtrain_only([inds_u_train, inds_i_train], y_train)

    print('After xtrain:')
    result = model.evaluate([inds_u_test, inds_i_test], y_test)
    print(result)