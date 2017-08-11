import script_chdir
import numpy as np
from evaluation.eval_script import evaluate_models_xval, EvalModel, print_results
from hybrid_model.dataset import get_dataset
from evaluation.evaluation import Evaluation
from evaluation.evaluation_metrics import TopN


np.set_printoptions(precision=4)

# Get dataset
dataset = get_dataset('ml100k')

models = []

# Item Average
from hybrid_model.models import BiasEstimatorCustom
model_type = BiasEstimatorCustom
config = {'include_user': False, 'include_item': True}
models.append(EvalModel(model_type.__name__, model_type, config))

# Bias Baseline
from hybrid_model.models import BiasEstimator
model_type = BiasEstimator
config = {}
models.append(EvalModel(model_type.__name__, model_type, config))


# SVDpp
from hybrid_model.models import SVDpp
model_type = SVDpp
config = {}
models.append(EvalModel(model_type.__name__, model_type, config))

# SigmoidUserAsymFactoring
from hybrid_model.models import SigmoidUserAsymFactoring
model_type = SigmoidUserAsymFactoring
config = {'implicit_thresh': 4.0, 'n_factors': 79,
          'reg_bias': 0.004770353622067247, 'reg_latent': 2.3618479038250382e-05}
models.append(EvalModel(model_type.__name__, model_type, config))

evaluater = Evaluation([TopN(100), ])
results = evaluate_models_xval(dataset, models, coldstart=False, evaluater=evaluater)

print_results(results)
