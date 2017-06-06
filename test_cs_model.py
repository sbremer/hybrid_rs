from eval_script import evaluate_models_single, evaluate_models_xval, print_results, EvalModel
from hybrid_model.evaluation import Evaluation
from hybrid_model import evaluation_metrics
from hybrid_model.dataset import get_dataset

"""
ml100k
------- BaselineBias
Combined Results:
=== Part full
rmse: 1.0118 ± 0.0006

------- AttributeBias
Combined Results:
=== Part full
rmse: 1.0102 ± 0.0012

------- AttributeBiasExperimental
Combined Results:
=== Part full
rmse: 1.0091 ± 0.0010

ml1m
------- BaselineBias
Combined Results:
=== Part full
rmse: 0.9823 ± 0.0002

------- AttributeBiasExperimental
Combined Results:
=== Part full
rmse: 0.9785 ± 0.0004
"""

# Get dataset
# dataset = get_dataset('ml100k')
dataset = get_dataset('ml1m')

models = []

# # Hybrid Model
# from hybrid_model.hybrid import HybridModel
# from hybrid_model.config import hybrid_config
# model_type = HybridModel
# config = hybrid_config
# models.append(EvalModel(model_type.__name__, model_type, config))

# # Bias Baseline
# from hybrid_model.baselines import BaselineBias
# model_type = BaselineBias
# config = dict(reg_bias=0.000003)
# models.append(EvalModel(model_type.__name__, model_type, config))

"""
l1
0.9115 @ n_factors=35, reg_bias=0.000001, reg_latent=0.000001
l2
0.9155 @ n_factors=35, reg_bias=0.00001, reg_latent=0.00001

"""

# # SVD
# from hybrid_model.baselines import BaselineSVD
# model_type = BaselineSVD
# config = dict(n_factors=35, reg_bias=0.00002, reg_latent=0.00002)
# models.append(EvalModel(model_type.__name__, model_type, config))

# SVD++
from hybrid_model.baselines import BaselineSVDpp
model_type = BaselineSVDpp
config = dict(n_factors=35, reg_bias=0.00001, reg_latent=0.00003, implicit_thresh=3.5)
models.append(EvalModel(model_type.__name__, model_type, config))

# from hybrid_model.baselines import AttributeBias
# model_type = AttributeBias
# config = dict(reg_att_bias=0.0015, reg_bias=0.00005)
# models.append(EvalModel(model_type.__name__, model_type, config))

# from hybrid_model.baselines import AttributeBiasExperimental
# model_type = AttributeBiasExperimental
# config = dict(reg_bias=0.000003, reg_att_bias=0.000005)
# models.append(EvalModel(model_type.__name__, model_type, config))

metrics = {'rmse': evaluation_metrics.Rmse(), 'prec@5': evaluation_metrics.Precision(5)}
evaluation = Evaluation(metrics)

# results = evaluate_models_xval(dataset, models, user_coldstart=True, n_entries=0, evaluation=evaluation)
# print('Coldstart0')
# print_results(results)
#
# results = evaluate_models_xval(dataset, models, user_coldstart=True, n_entries=10, evaluation=evaluation)
# print('Coldstart10')
# print_results(results)
#
results = evaluate_models_xval(dataset, models, user_coldstart=True, n_entries=50, evaluation=evaluation)
print('Coldstart30')
print_results(results)

"""
ml1m
0: Bias 0.9793 AttE 0.9739 SVD++ 0.9947 SVD 0.9790
10: AttE 0.9371 SVD 0.9301
20: Bias 0.9261 AttE 0.9102 SVD++ 0.9420 SVD 0.9104
30: AttE 0.8995 SVD 0.9029 SVD++ 0.8988
50: AttE 0.8948 SVD 0.8999

"""

# results = evaluate_models_xval(dataset, models, user_coldstart=False, evaluation=evaluation)
# print('Normal')
# print_results(results)