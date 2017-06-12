from hybrid_model.evaluation import Evaluation

from eval_script import evaluate_models_xval, print_results, EvalModel
from evaluation import evaluation_metrics
from hybrid_model.dataset import get_dataset

dataset = get_dataset('ml100k')

models = []

# Bias Baseline
from hybrid_model.baselines import BaselineBias
model_type = BaselineBias
config = dict(reg_bias=0.000003)
models.append(EvalModel(model_type.__name__, model_type, config))

# SVD++
from hybrid_model.baselines import BaselineSVDpp
model_type = BaselineSVDpp
config = dict(n_factors=35, reg_bias=0.00001, reg_latent=0.00003, implicit_thresh=3.5)
models.append(EvalModel(model_type.__name__, model_type, config))

# AttributeBiasExperimental
from hybrid_model.baselines import AttributeBiasExperimental
model_type = AttributeBiasExperimental
config = dict(reg_bias=0.000003, reg_att_bias=0.000005)
models.append(EvalModel(model_type.__name__, model_type, config))

metrics = {'rmse': evaluation_metrics.Rmse(), 'prec@5': evaluation_metrics.Precision(5)}
evaluation = Evaluation(metrics)

results = evaluate_models_xval(dataset, models, coldstart=False, evaluation=evaluation)
print_results(results)
