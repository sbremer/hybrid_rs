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
rmse: 0.8995 ± 0.0031
MD:
Combined Results:
=== Part full
rmse: 0.9266 ± 0.0044
Hybrid after xtrain:
CF:
Combined Results:
=== Part full
rmse: 0.8967 ± 0.0021
MD:
Combined Results:
=== Part full
rmse: 0.9241 ± 0.0042

"""

results = evaluate_models_xval(dataset, models, user_coldstart=False)
print('Normal')
print_results(results)
