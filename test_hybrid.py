from evaluation.eval_script import evaluate_models_xval, evaluate_models_single, evaluate_hybrid_single, evaluate_hybrid_xval, print_results, print_hybrid_results, EvalModel
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
=== Part full
rmse: 0.9004 ± 0.0027
MD:
=== Part full
rmse: 0.9265 ± 0.0038

Hybrid after xtrain:
CF:
=== Part full
rmse: 0.8964 ± 0.0027
MD:
=== Part full
rmse: 0.9245 ± 0.0039

Coldstart User
------- HybridModel
Hybrid before xtrain:
CF:
=== Part full
rmse: 1.0873 ± 0.0216
MD:
=== Part full
rmse: 1.0180 ± 0.0252

Hybrid after xtrain:
CF:
=== Part full
rmse: 1.0244 ± 0.0249
MD:
=== Part full
rmse: 1.0173 ± 0.0254

Coldstart Item
------- HybridModel
Hybrid before xtrain:
CF:
=== Part full
rmse: 1.1338
MD:
=== Part full
rmse: 1.0467

Hybrid after xtrain:
CF:
=== Part full
rmse: 1.0620
MD:
=== Part full
rmse: 1.0452
"""

# results = evaluate_models_xval(dataset, models, coldstart=False)
# print('Normal')
# print_results(results)

# results = evaluate_models_xval(dataset, models, coldstart=True, cs_type='user')
# print('Coldstart User')
# print_results(results)

results = evaluate_hybrid_xval(dataset, hybrid_config, coldstart=True, cs_type='item', n_entries=0)
print('Coldstart Item')
print_hybrid_results(results)
