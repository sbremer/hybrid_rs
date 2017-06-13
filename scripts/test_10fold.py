from evaluation.eval_script import evaluate_models_xval, print_results, EvalModel
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

results = evaluate_models_xval(dataset, models, coldstart=False, n_fold=10)
print('Normal 10-fold')
print_results(results)

"""
Normal 10-fold
------- HybridModel
Hybrid before xtrain:
CF:
Combined Results:
=== Part full
rmse: 0.8952 ± 0.0054
MD:
Combined Results:
=== Part full
rmse: 0.9258 ± 0.0047

Hybrid after xtrain:
CF:
Combined Results:
=== Part full
rmse: 0.8906 ± 0.0047
MD:
Combined Results:
=== Part full
rmse: 0.9207 ± 0.0042
"""
