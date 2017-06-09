import numpy as np

np.random.seed(0)

from evaluation.evaluation import Evaluation
from hybrid_model.hybrid import HybridModel
from hybrid_model.hybrid import HybridConfig
from hybrid_model import index_sampler
from hybrid_model import transform
from hybrid_model import models
from hybrid_model.dataset import get_dataset
from util import kfold

user_coldstart = False
n_entries = 0
n_fold = 5

hybrid_config = HybridConfig(
    model_type_cf=models.SVDpp,
    model_config_cf={'n_factors': 40, 'reg_bias': 0.00005, 'reg_latent': 0.00003},
    model_type_md=models.AttributeBiasExperimental,
    model_config_md={'reg_bias': 0.0001, 'reg_att_bias': 0.0003},
    opt_cf_init='nadam',
    opt_md_init='nadam',
    opt_cf_xtrain='adagrad',
    opt_md_xtrain='adagrad',
    batch_size_init_cf=256,
    batch_size_init_md=1024,
    batch_size_xtrain_cf=256,
    batch_size_xtrain_md=1024,
    val_split_init=0.05,
    val_split_xtrain=0.05,
    index_sampler=index_sampler.IndexSamplerUserItembased,
    xtrain_epochs=6,
    xtrain_data_shuffle=False,
    transformation=transform.TransformationLinear
)

evaluation = Evaluation()
dataset = get_dataset('ml100k')

(inds_u, inds_i, y, users_features, items_features) = dataset.data

if user_coldstart:
    if n_entries == 0:
        fold = kfold.kfold_entries(n_fold, inds_u)
    else:
        fold = kfold.kfold_entries_plus(n_fold, inds_u, n_entries)
else:
    fold = kfold.kfold(n_fold, inds_u)

fold = list(fold)
fold = [fold[3]]

for xval_train, xval_test in fold:

    # Dataset training
    inds_u_train = inds_u[xval_train]
    inds_i_train = inds_i[xval_train]
    y_train = y[xval_train]
    n_train = len(y_train)

    # Dataset testing
    inds_u_test = inds_u[xval_test]
    inds_i_test = inds_i[xval_test]
    y_test = y[xval_test]

    train = ([inds_u_train, inds_i_train], y_train)
    test = ([inds_u_test, inds_i_test], y_test)

    hybrid_model = HybridModel(users_features, items_features, hybrid_config, verbose=2)

    hybrid_model.fit_init(*train)
    result_before_x = evaluation.evaluate_hybrid(hybrid_model, *test)

    print('After initial:')
    print(result_before_x)

    hybrid_model.setup_cross_training()

    for e in range(hybrid_config.xtrain_epochs):
        hybrid_model.fit_cross_epoch()
        result = evaluation.evaluate_hybrid(hybrid_model, *test)

        print('After epoch', e+1)
        print(result)
