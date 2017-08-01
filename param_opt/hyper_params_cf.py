import numpy as np
np.random.seed(0)

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Local imports
import script_chdir
from evaluation.evaluation import Evaluation
from hybrid_model import dataset
from util import kfold

# Get dataset
ds = dataset.get_dataset('ml100k')
(inds_u, inds_i, y, users_features, items_features) = ds.data

n_fold = 5
folds_normal = list(kfold.kfold(n_fold, inds_u))

evaluater = Evaluation()

at = 0

from hybrid_model.models import SigmoidUserAsymFactoring
model_type = SigmoidUserAsymFactoring


def test(config):
    global at
    at += 1
    print('At:', at)

    results = evaluater.get_results_class()

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

        # Build config
        # config = model_type(**config)
        # model = model_type(users_features, items_features, config)
        model = model_type(ds.n_users, ds.n_items, config)

        model.fit(*train)

        result = evaluater.evaluate(model, *test)
        results.add(result)

    return {'loss': results.rmse(), 'status': STATUS_OK, 'param': config}


# Hybrid config for mercateo
config_space = {'n_factors': hp.choice('n_factors', np.arange(5, 101)),
                'reg_bias': hp.loguniform('reg_bias_cf', -15, -4),
                'reg_latent': hp.loguniform('reg_latent', -15, -4),
                'implicit_thresh': hp.choice('implicit_thresh', [1.0, 2.0, 3.0, 4.0, 5.0]),
}

trials = Trials()
best = fmin(test, config_space, algo=tpe.suggest, max_evals=300, trials=trials)
print('Best RMSE:', trials.best_trial['result']['loss'])
print(trials.best_trial['result']['param'])
