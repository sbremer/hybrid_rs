import pickle
import numpy as np

np.random.seed(0)

# Local imports
from hybrid_model.hybrid import HybridModel, HybridConfig
from hybrid_model import transform
from hybrid_model.index_sampler import IndexSampler2, IndexSamplerUserbased
from hybrid_model.evaluation import EvaluationResults
from hybrid_model.evaluation_parting import stats_user, stats_item
import util

(inds_u, inds_i, y, users_features, items_features) = pickle.load(open('data/ml100k.pickle', 'rb'))

n_users, n_users_features = users_features.shape
n_items, n_items_features = items_features.shape

# Crossvalidation
n_fold = 5
user_coldstart = False
if user_coldstart:
    kfold = util.kfold_entries(n_fold, inds_u)
    # kfold = util.kfold_entries_plus(n_fold, inds_u, 34)
else:
    kfold = util.kfold(n_fold, inds_u)

kfold = list(kfold)

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

    model.fit_init_only([inds_u_train, inds_i_train], y_train)
    result_before_xtrain = model.evaluate([inds_u_test, inds_i_test], y_test)

    model.fit_xtrain_only([inds_u_train, inds_i_train], y_train)
    result_after_xtrain = model.evaluate([inds_u_test, inds_i_test], y_test)

    results_before_xtrain.add(result_before_xtrain)
    results_after_xtrain.add(result_after_xtrain)

print('Before xtrain:')
print(results_before_xtrain)

print('After xtrain:')
print(results_after_xtrain)

for b in range(10):
    print('Avg #ratings/user for user bin {}/{}: {}'.format(b + 1, 10, np.mean(stats_user[b])))

for b in range(10):
    print('Avg #ratings/item for item bin {}/{}: {}'.format(b + 1, 10, np.mean(stats_item[b])))
