import numpy as np
np.random.seed(0)

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import gc
import subprocess

# Local imports
# import script_chdir
from hybrid_model.hybrid import HybridModel
from evaluation.evaluation import Evaluation
from hybrid_model import dataset
from util import kfold

# Get dataset
ds = dataset.get_dataset('ml100k')
(inds_u, inds_i, y, users_features, items_features) = ds.data

n_fold = 5
folds_normal = list(kfold.kfold(n_fold, inds_u))

# Choose metric to optimize against and whether to minimize or maximize
metric = 'TopNAURC(k=100)'
metric_factor = -1.0  # 1.0 -> Minimize (For error like RMSE) / -1.0 -> Maximize (for Precision/Recall and such)

at = 0


def test(config):
    global at
    at += 1
    print('At:', at)

    evaluater = Evaluation()
    results = evaluater.get_results_class()

    # Build config
    hybrid_config = HybridConfig(**config)

    for xval_train, xval_test in folds_normal:

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

        model = HybridModel(users_features, items_features, hybrid_config)

        model.fit_init(*train)
        model.fit_cross()

        result = evaluater.evaluate(model, *train, *test)
        results.add(result)

        del model

        for i in range(3):
            gc.collect()

    return {'loss': metric_factor * results.mean(metric), 'status': STATUS_OK, 'param': config}


from hybrid_model.hybrid import HybridConfig
from hybrid_model import index_sampler
from hybrid_model import transform
from hybrid_model import models

# Hybrid config for mercateo
config_space = dict(
    model_type_cf=models.SigmoidUserAsymFactoring,
    model_config_cf={'implicit_thresh': 4.0,
                     'implicit_thresh_crosstrain': hp.uniform('implicit_thresh_crosstrain', 3.0, 5.0),
                     'n_factors': 79,
                     'reg_bias': 0.004770353622067247,
                     'reg_latent': 2.3618479038250382e-05},
    model_type_md=models.AttributeBiasAdvanced,
    model_config_md={'reg_att_bias': 6.578729437598415e-07, 'reg_bias': 6.842025959062749e-07},
    batch_size_cf=hp.choice('batch_size_cf', [128, 256, 512, 1024, 2048]),
    batch_size_md=hp.choice('batch_size_md', [128, 256, 512, 1024, 2048]),
    val_split_init=0.05,
    val_split_xtrain=0.05,
    index_sampler=index_sampler.IndexSamplerUserItembased,
    index_sampler_config={'f_cf': hp.uniform('f_cf', 0.05, 0.5),
                          'min_ratings_user': hp.choice('min_ratings_user', np.arange(5, 40)),
                          'f_user': hp.uniform('f_user', 0.5, 4.0),
                          'min_ratings_item': hp.choice('min_ratings_item', np.arange(5, 40)),
                          'f_item': hp.uniform('f_item', 0.5, 4.0)},
    xtrain_epochs=hp.choice('xtrain_epochs', np.arange(1, 6)),
    xtrain_data_shuffle=True,
    cutoff_user=hp.choice('cutoff_user', np.arange(0, 30)),
    cutoff_item=hp.choice('cutoff_item', np.arange(0, 30)),
    transformation=transform.TransformationLinear
)

trials = Trials()
best = fmin(test, config_space, algo=tpe.suggest, max_evals=300, trials=trials)
print('Best {}: {}'.format(metric, metric_factor * trials.best_trial['result']['loss']))
print(trials.best_trial['result']['param'])

# Best TopNAURC(k=100): 0.8954019730955192
# {'batch_size_cf': 512, 'batch_size_md': 1024, 'cutoff_item': 1, 'cutoff_user': 2, 'index_sampler': <class 'hybrid_model.index_sampler.IndexSamplerUserItembased'>, 'index_sampler_config': {'f_cf': 0.2851797465552042, 'f_item': 0.8689756294257732, 'f_user': 2.4239659839472605, 'min_ratings_item': 19, 'min_ratings_user': 25}, 'model_config_cf': {'implicit_thresh': 4.0, 'implicit_thresh_crosstrain': 4.390126327991918, 'n_factors': 79, 'reg_bias': 0.004770353622067247, 'reg_latent': 2.3618479038250382e-05, 'transformation': <class 'hybrid_model.transform.TransformationLinear'>}, 'model_config_md': {'reg_att_bias': 6.578729437598415e-07, 'reg_bias': 6.842025959062749e-07, 'transformation': <class 'hybrid_model.transform.TransformationLinear'>}, 'model_type_cf': <class 'hybrid_model.models.sigmoid_user_asymfactoring.SigmoidUserAsymFactoring'>, 'model_type_md': <class 'hybrid_model.models.attributebias_advanced.AttributeBiasAdvanced'>, 'transformation': <class 'hybrid_model.transform.TransformationLinear'>, 'val_split_init': 0.05, 'val_split_xtrain': 0.05, 'xtrain_data_shuffle': True, 'xtrain_epochs': 3}
