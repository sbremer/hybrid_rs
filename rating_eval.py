import pickle
import numpy as np

# np.random.seed(0)

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

n_fold = 5
rep_xval = 3

x = []
y_mf = []
y_cs = []

for ratings_per_user in range(0, 61, 5):

    results_before_xtrain = EvaluationResults()

    for _ in range(rep_xval):

        kfold = util.kfold_entries_plus(n_fold, inds_u, ratings_per_user)

        # Create config for model
        hybrid_config = HybridConfig(
            n_factors=40,
            reg_bias_mf=0.00005,
            reg_latent=0.00004,#2
            reg_bias_cs=0.0001,
            reg_att_bias=0.0015,
            implicit_thresh_init=0.7,
            implicit_thresh_xtrain=0.8,
            opt_mf_init='adadelta',
            opt_cs_init='adadelta',
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

            results_before_xtrain.add(result_before_xtrain)

    rmse_mf = results_before_xtrain.mean_rmse_mf()
    rmse_cs = results_before_xtrain.mean_rmse_cs()
    print('For {} ratings: MF {}  CS {}'.format(ratings_per_user, rmse_mf, rmse_cs))

    x.append(ratings_per_user)
    y_mf.append(rmse_mf)
    y_cs.append(rmse_cs)

print('x =', x)
print('y_mf =', y_mf)
print('y_cs =', y_cs)
