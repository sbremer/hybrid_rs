from eval_script import evaluate_models_single, evaluate_models_xval, print_results, EvalModel
from hybrid_model.evaluation import Evaluation
from hybrid_model import evaluation_metrics
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

results = evaluate_models_xval(dataset, models, user_coldstart=False)
print('Normal')
print_results(results)

"""
Normal
------- HybridModel
Hybrid before xtrain:
CF:
Combined Results:
=== Part full
rmse: 0.9036 ± 0.0030
MD:
Combined Results:
=== Part full
rmse: 0.9308 ± 0.0010

Hybrid after xtrain:
CF:
Combined Results:
=== Part full
rmse: 0.8936 ± 0.0004
MD:
Combined Results:
=== Part full
rmse: 0.9276 ± 0.0007

Cold-start
------- HybridModel
Hybrid before xtrain:
CF:
Combined Results:
=== Part full
rmse: 1.0653 ± 0.0022
MD:
Combined Results:
=== Part full
rmse: 1.0097 ± 0.0014

Hybrid after xtrain:
CF:
Combined Results:
=== Part full
rmse: 1.0099 ± 0.0026
MD:
Combined Results:
=== Part full
rmse: 1.0144 ± 0.0024
"""
