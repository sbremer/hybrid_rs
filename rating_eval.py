import pickle
import numpy as np

# np.random.seed(0)

# Local imports
from hybrid_model.hybrid import HybridModel, HybridConfig
from hybrid_model import transform
from hybrid_model.index_sampler import IndexSampler2, IndexSamplerUserbased
from hybrid_model.evaluation import EvaluationResults, EvaluationResultsModel
from hybrid_model.evaluation_parting import stats_user, stats_item
import util

from hybrid_model.baselines import BaselineSVD, BiasEstimator, BaselineBias
from hybrid_model.callbacks_custom import EarlyStoppingBestVal

(inds_u, inds_i, y, users_features, items_features) = pickle.load(open('data/ml100k.pickle', 'rb'))

n_users, n_users_features = users_features.shape
n_items, n_items_features = items_features.shape

callbacks = [EarlyStoppingBestVal('val_loss', patience=10)]

n_fold = 5
rep_xval = 3

x = []
y_mf = []
y_cs = []
y_mf_baseline = []
y_cbias_baseline = []
y_bias_baseline = []

for ratings_per_user in range(0, 61, 5):

    results_before_xtrain = EvaluationResults()

    results_mf = EvaluationResultsModel()
    results_cbias_baseline = EvaluationResultsModel()
    results_bias_baseline = EvaluationResultsModel()

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

            # model = BiasEstimator(n_users, n_items)
            # model.fit([inds_u_train, inds_i_train], y_train)
            # result = model.evaluate([inds_u_test, inds_i_test], y_test)
            # results_cbias_baseline.add(result)

            model = BaselineBias(n_users, n_items, transformation=transform.TransformationLinear())
            model.fit([inds_u_train, inds_i_train], y_train, batch_size=512, epochs=200,
                          validation_split=0.05, verbose=0, callbacks=callbacks)

            result = model.evaluate([inds_u_test, inds_i_test], y_test)
            results_bias_baseline.add(result)

            # model = BaselineSVD(n_users, n_items, transformation=transform.TransformationLinear())
            #
            # model.fit([inds_u_train, inds_i_train], y_train, batch_size=512, epochs=200,
            #               validation_split=0.05, verbose=0, callbacks=callbacks)
            #
            # result_mf = model.evaluate([inds_u_test, inds_i_test], y_test)
            # results_mf.add(result_mf)

            # Create model
            # model = HybridModel(users_features, items_features, hybrid_config, verbose=0)
            #
            # model.fit_init_only([inds_u_train, inds_i_train], y_train)
            # model.fit_xtrain_only([inds_u_train, inds_i_train], y_train)
            # result = model.evaluate([inds_u_test, inds_i_test], y_test)
            #
            # results_before_xtrain.add(result)

    # rmse_mf = results_before_xtrain.mean_rmse_mf()
    # rmse_cs = results_before_xtrain.mean_rmse_cs()
    # print('For {} ratings: MF {}  CS {}'.format(ratings_per_user, rmse_mf, rmse_cs))

    # rmse_mf_baseline = results_mf.mean('rmse')
    # print('For {} ratings: MF_Baseline {}'.format(ratings_per_user, rmse_mf_baseline))

    # rmse_cbias_baselne = results_cbias_baseline.mean('rmse')
    # print('For {} ratings: cbias_baseline {}'.format(ratings_per_user, rmse_cbias_baselne))

    rmse_bias_baselne = results_bias_baseline.mean('rmse')
    print('For {} ratings: bias_baseline {}'.format(ratings_per_user, rmse_bias_baselne))

    x.append(ratings_per_user)
    # y_mf.append(rmse_mf)
    # y_cs.append(rmse_cs)
    # y_mf_baseline.append(rmse_mf_baseline)
    # y_cbias_baseline.append(rmse_cbias_baselne)
    y_bias_baseline.append(rmse_bias_baselne)

print('x =', x)
# print('y_mf_x =', y_mf)
# print('y_cs_x =', y_cs)
# print('y_mf_baseline =', y_mf_baseline)
# print('y_cbias_baseline =', y_cbias_baseline)
print('y_bias_baseline =', y_bias_baseline)
