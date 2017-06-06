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
