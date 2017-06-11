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
n_inds_from_md = 13696
n_inds_from_md = 13831
n_inds_from_md = 13835
n_inds_from_md = 14016
n_inds_from_md = 13747
Normal
------- HybridModel
Hybrid before xtrain:
CF:
Combined Results:
=== Part full
rmse: 0.8983 ± 0.0019
MD:
Combined Results:
=== Part full
rmse: 0.9286 ± 0.0047

Hybrid after xtrain:
CF:
Combined Results:
=== Part full
rmse: 0.8964 ± 0.0023
MD:
Combined Results:
=== Part full
rmse: 0.9231 ± 0.0039

n_inds_from_md = 27172
n_inds_from_md = 27511
n_inds_from_md = 27321
n_inds_from_md = 27314
n_inds_from_md = 27161
Coldstart
------- HybridModel
Hybrid before xtrain:
CF:
Combined Results:
=== Part full
rmse: 1.0893 ± 0.0083
MD:
Combined Results:
=== Part full
rmse: 1.0201 ± 0.0130

Hybrid after xtrain:
CF:
Combined Results:
=== Part full
rmse: 1.0213 ± 0.0124
MD:
Combined Results:
=== Part full
rmse: 1.0172 ± 0.0129

"""

results = evaluate_models_xval(dataset, models, user_coldstart=False)
print('Normal')
print_results(results)

results = evaluate_models_xval(dataset, models, user_coldstart=True)
print('Coldstart')
print_results(results)
