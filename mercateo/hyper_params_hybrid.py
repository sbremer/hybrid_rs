import numpy as np
np.random.seed(0)

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Local imports
import script_chdir
from hybrid_model.hybrid import HybridModel
from evaluation.evaluation import Evaluation
from hybrid_model import dataset
from util import kfold

# Add mercateo dataset (which should not be included in the open source repo on github)
dataset._datasets['mercateo'] = 'data/mercateo'

# Get dataset
dataset = dataset.get_dataset('mercateo')
(inds_u, inds_i, y, users_features, items_features) = dataset.data

n_fold = 5
folds = list(kfold.kfold(n_fold, inds_u))

evaluater = Evaluation()

at = 0


def test(config):
    global at
    at += 1
    print('At:', at)

    for xval_train, xval_test in folds:

        # Dataset training
        inds_u_train = inds_u[xval_train]
        inds_i_train = inds_i[xval_train]
        y_train = y[xval_train]

        # Dataset testing
        inds_u_test = inds_u[xval_test]
        inds_i_test = inds_i[xval_test]
        y_test = y[xval_test]

        train = ([inds_u_train, inds_i_train], y_train)
        test = ([inds_u_test, inds_i_test], y_test)

        # Build config
        config = HybridConfig(**config)
        model = HybridModel(users_features, items_features, config)

        model.fit_init(*train)
        model.fit_cross()

        result_hybrid = evaluater.evaluate(model, *test)

        return {'loss': result_hybrid.rmse(), 'status': STATUS_OK, 'param': config}



from hybrid_model.hybrid import HybridConfig
from hybrid_model import index_sampler
from hybrid_model import transform
from hybrid_model import models

# Hybrid config for mercateo
config_space = dict(
    model_type_cf=models.SigmoidUserAsymFactoring,
    model_config_cf={'n_factors': 40, 'reg_bias': 0.000005, 'reg_latent': 0.00003,
                     'implicit_thresh': 4.0, 'implicit_thresh_crosstrain': 4.5, 'optimizer': 'adagrad'},
    model_type_md=models.AttributeBiasAdvanced,
    model_config_md={'reg_bias': 0.0002, 'reg_att_bias': 0.0004, 'optimizer': 'adagrad'},
    batch_size_cf=hp.choice('batch_size_cf', [128, 256, 512, 1024, 2048]),
    batch_size_md=hp.choice('batch_size_md', [128, 256, 512, 1024, 2048]),
    val_split_init=0.05,
    val_split_xtrain=0.05,
    index_sampler=index_sampler.IndexSamplerUserItembased,
    index_sampler_config={'f_cf': hp.uniform('f_cf', 0.05, 0.5),
                          'min_ratings_user': hp.choice('min_ratings_user', np.arange(1, 16)),
                          'f_user': hp.uniform('f_user', 0.5, 4.0),
                          'min_ratings_item': hp.choice('min_ratings_item', np.arange(1, 16)),
                          'f_item': hp.uniform('f_item', 0.5, 4.0)},
    xtrain_epochs=4,
    xtrain_data_shuffle=True,
    cutoff_user=hp.choice('cutoff_user', np.arange(1, 16)),
    cutoff_item=hp.choice('cutoff_item', np.arange(1, 16)),
    transformation=transform.TransformationLinear
)

trials = Trials()
best = fmin(test, config_space, algo=tpe.suggest, max_evals=300, trials=trials)
print('Best: ')
print(trials.best_trial['result']['param'])
