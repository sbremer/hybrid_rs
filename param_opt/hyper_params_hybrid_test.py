import subprocess
import numpy as np
np.random.seed(0)

# Local imports
from hybrid_model.hybrid import HybridModel, HybridConfig
from evaluation.evaluation import Evaluation
from hybrid_model import dataset
from util import kfold

from hybrid_model import index_sampler
from hybrid_model import transform
from hybrid_model import models

# Get dataset
ds = dataset.get_dataset('ml100k')
(inds_u, inds_i, y, users_features, items_features) = ds.data

n_fold = 5
folds_normal = list(kfold.kfold(n_fold, inds_u))

# Choose metric to optimize against and whether to minimize or maximize
metric = 'TopNAURC(k=100)'
metric_factor = -1.0  # 1.0 -> Minimize (For error like RMSE) / -1.0 -> Maximize (for Precision/Recall and such)


def test(config):

    config['model_type_cf'] = models.SigmoidUserAsymFactoring
    config['model_type_md'] = models.AttributeBiasAdvanced
    config['index_sampler'] = index_sampler.IndexSamplerUserItembased
    config['transformation'] = transform.TransformationLinear

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

    return metric_factor * results.mean(metric)