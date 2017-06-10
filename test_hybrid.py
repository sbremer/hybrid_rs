from eval_script import evaluate_models_xval, evaluate_models_single, print_results, EvalModel
from hybrid_model.dataset import get_dataset

# Get dataset
dataset = get_dataset('ml100k')
# dataset = get_dataset('ml1m')

models = []

# Hybrid Model
from hybrid_model.hybrid import HybridModel
from hybrid_model.config import hybrid_config

model_type = HybridModel
config = hybrid_config
models.append(EvalModel(model_type.__name__, model_type, config))

"""
Normal
------- HybridModel
Hybrid before xtrain:
CF:
Combined Results:
=== Part full
rmse: 0.8992 ± 0.0035
MD:
Combined Results:
=== Part full
rmse: 0.9262 ± 0.0041

Hybrid after xtrain:
CF:
Combined Results:
=== Part full
rmse: 0.8960 ± 0.0024
MD:
Combined Results:
=== Part full
rmse: 0.9239 ± 0.0044


Normal
------- HybridModel
Hybrid before xtrain:
CF:
Combined Results:
=== Part full
rmse: 0.8996 ± 0.0029
MD:
Combined Results:
=== Part full
rmse: 0.9259 ± 0.0040

Hybrid after xtrain:
CF:
Combined Results:
=== Part full
rmse: 0.8962 ± 0.0017
MD:
Combined Results:
=== Part full
rmse: 0.9241 ± 0.0042

n_inds_from_md = 27391
n_inds_from_md = 27454
n_inds_from_md = 27242
n_inds_from_md = 27537
n_inds_from_md = 26849
Coldstart
------- HybridModel
Hybrid before xtrain:
CF:
Combined Results:
=== Part full
rmse: 1.0907 ± 0.0396
MD:
Combined Results:
=== Part full
rmse: 1.0204 ± 0.0489

Hybrid after xtrain:
CF:
Combined Results:
=== Part full
rmse: 1.0232 ± 0.0491
MD:
Combined Results:
=== Part full
rmse: 1.0192 ± 0.0474

"""

results = evaluate_models_xval(dataset, models, user_coldstart=False)
print('Normal')
print_results(results)

results = evaluate_models_xval(dataset, models, user_coldstart=True)
print('Coldstart')
print_results(results)
