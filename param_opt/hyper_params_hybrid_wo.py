import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import subprocess

# Choose metric to optimize against and whether to minimize or maximize
metric = 'TopNRecall(k=100)'
metric_factor = -1.0  # 1.0 -> Minimize (For error like RMSE) / -1.0 -> Maximize (for Precision/Recall and such)

at = 0


def test_wrapper(config):

    global at
    at += 1
    print('At:', at)

    cmd = 'config = ' + str(config) + ';'
    cmd += 'from param_opt.hyper_params_hybrid_test import test;'
    cmd += 'print(test(config), end="")'
    ret = subprocess.check_output(['python', '-c', cmd], stderr=subprocess.PIPE)

    loss = float(ret)

    return {'loss': loss, 'status': STATUS_OK, 'param': config}


# from hybrid_model import index_sampler
# from hybrid_model import transform
# from hybrid_model import models

# Hybrid config for mercateo
config_space = dict(
    # model_type_cf=models.SigmoidUserAsymFactoring,
    model_config_cf={'implicit_thresh': 4.0,
                     'implicit_thresh_crosstrain': hp.uniform('implicit_thresh_crosstrain', 3.0, 5.0),
                     'n_factors': 79,
                     'reg_bias': 0.004770353622067247,
                     'reg_latent': 2.3618479038250382e-05},
    # model_type_md=models.AttributeBiasAdvanced,
    model_config_md={'reg_att_bias': 6.578729437598415e-07, 'reg_bias': 6.842025959062749e-07},
    batch_size_cf=hp.choice('batch_size_cf', [128, 256, 512, 1024, 2048]),
    batch_size_md=hp.choice('batch_size_md', [128, 256, 512, 1024, 2048]),
    val_split_init=0.05,
    val_split_xtrain=0.05,
    # index_sampler=index_sampler.IndexSamplerUserItembased,
    index_sampler_config={'f_cf': hp.uniform('f_cf', 0.05, 0.5),
                          'min_ratings_user': hp.choice('min_ratings_user', np.arange(5, 40)),
                          'f_user': hp.uniform('f_user', 0.5, 4.0),
                          'min_ratings_item': hp.choice('min_ratings_item', np.arange(5, 40)),
                          'f_item': hp.uniform('f_item', 0.5, 4.0)},
    xtrain_epochs=hp.choice('xtrain_epochs', np.arange(1, 6)),
    xtrain_data_shuffle=True,
    cutoff_user=hp.choice('cutoff_user', np.arange(0, 30)),
    cutoff_item=hp.choice('cutoff_item', np.arange(0, 30)),
    # transformation=transform.TransformationLinear,
)

trials = Trials()
best = fmin(test_wrapper, config_space, algo=tpe.suggest, max_evals=300, trials=trials)
print('Best {}: {}'.format(metric, metric_factor * trials.best_trial['result']['loss']))
print(trials.best_trial['result']['param'])

# Best TopNRecall(k=100): 0.901852431977
# {'batch_size_cf': 1024, 'batch_size_md': 2048, 'cutoff_item': 0, 'cutoff_user': 10, 'index_sampler_config': {'f_cf': 0.1928720053014314, 'f_item': 0.5082244606009562, 'f_user': 0.9913654219276606, 'min_ratings_item': 8, 'min_ratings_user': 24}, 'model_config_cf': {'implicit_thresh': 4.0, 'implicit_thresh_crosstrain': 4.58158909923149, 'n_factors': 79, 'reg_bias': 0.004770353622067247, 'reg_latent': 2.3618479038250382e-05}, 'model_config_md': {'reg_att_bias': 6.578729437598415e-07, 'reg_bias': 6.842025959062749e-07}, 'val_split_init': 0.05, 'val_split_xtrain': 0.05, 'xtrain_data_shuffle': True, 'xtrain_epochs': 5}

