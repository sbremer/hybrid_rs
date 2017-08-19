import numpy as np
np.random.seed(0)

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Local imports
import script_chdir
from evaluation.evaluation import Evaluation
from hybrid_model import dataset
from util import kfold, timing

# Get dataset
ds = dataset.get_dataset('ml100k')
(inds_u, inds_i, y, users_features, items_features) = ds.data

n_fold = 5
folds_item = list(kfold.kfold_entries(n_fold, inds_i))

# Choose metric to optimize against and whether to minimize or maximize
metric = 'TopNAURC(k=100)'
metric_factor = -1.0  # 1.0 -> Minimize (For error like RMSE) / -1.0 -> Maximize (for Precision/Recall and such)

evaluater = Evaluation()

at = 0

from hybrid_model.models import AttributeBiasLight
model_type = AttributeBiasLight


def test(config):
    global at
    at += 1
    print('At:', at)

    results_item = evaluater.get_results_class()

    for xval_train, xval_test in folds_item:

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

        model = model_type(users_features, items_features, config)

        with timing.Timer() as t:
            model.fit(*train)

        result = evaluater.evaluate(model, *train, *test)
        result.results['runtime'] = t.interval
        results_item.add(result)

    r = results_item.mean(metric)

    return {'loss': metric_factor * r, 'status': STATUS_OK, 'param': config}


# Hybrid config for mercateo
config_space = {'reg_bias': hp.loguniform('reg_bias', -15, -4),
                'reg_att_bias': hp.loguniform('reg_att_bias', -15, -4),
                }

trials = Trials()
best = fmin(test, config_space, algo=tpe.suggest, max_evals=100, trials=trials)
print('Best {}: {}'.format(metric, metric_factor * trials.best_trial['result']['loss']))
print(trials.best_trial['result']['param'])

# Best TopNAURC(k=100): 0.586333967320098
# {'reg_att_bias': 4.156321415552967e-07, 'reg_bias': 0.01791415395580668}
