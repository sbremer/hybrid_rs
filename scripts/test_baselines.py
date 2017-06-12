import scripts
from evaluation.evaluation import Evaluation
from eval_script import evaluate_models_xval, print_results, EvalModel
from evaluation import evaluation_metrics
from hybrid_model.dataset import get_dataset

# Get dataset
dataset = get_dataset('ml100k')
# dataset = get_dataset('ml1m')

models = []

# Bias Baseline
from hybrid_model.models import BiasEstimator
model_type = BiasEstimator
config = {}
models.append(EvalModel(model_type.__name__, model_type, config))

# SVD
from hybrid_model.models import SVD
model_type = SVD
config = {}
models.append(EvalModel(model_type.__name__, model_type, config))

results = evaluate_models_xval(dataset, models, coldstart=False)
print('Normal')
print_results(results)

results = evaluate_models_xval(dataset, models, coldstart=True)
print('Coldstart User')
print_results(results)

"""
Normal
------- BiasEstimator
Combined Results:
=== Part full
rmse: 0.9417 ± 0.0048

------- SVD
Combined Results:
=== Part full
rmse: 0.9250 ± 0.0044

Coldstart User
------- BiasEstimator
=== Part full
rmse: 1.0214 ± 0.0248

------- SVD
=== Part full
rmse: 1.0340 ± 0.0234
"""
