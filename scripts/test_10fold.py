import scripts
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
=== Part full
rmse: 0.8908 ± 0.0044  prec@5: 0.8756 ± 0.0035

------- HybridModel_SVDpp
=== Part full
rmse: 0.8953 ± 0.0048  prec@5: 0.8747 ± 0.0031

------- HybridModel_AttributeBiasExperimental
=== Part full
rmse: 0.9235 ± 0.0041  prec@5: 0.8633 ± 0.0043
"""
