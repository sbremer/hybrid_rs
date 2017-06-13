from evaluation.eval_script import evaluate_models_xval, print_hybrid_results, EvalModel
from evaluation import evaluation
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

evaluater = evaluation.Evaluation(evaluation.metrics_rmse_prec, evaluation.get_parting_all(10))

results = evaluate_models_xval(dataset, hybrid_config, coldstart=False, evaluater=evaluater, n_fold=10, repeat=3)
print_hybrid_results(results)
