from evaluation.eval_script import evaluate_models_single, print_results, EvalModel
from hybrid_model.dataset import get_dataset

"""
------- HybridModel
Hybrid before xtrain:
MF:
=== Part full
rmse: 0.9055
CS:
=== Part full
rmse: 0.9353
Hybrid after xtrain:
MF:
=== Part full
rmse: 0.8923
CS:
=== Part full
rmse: 0.9340
------- BaselineSVDpp
=== Part full
rmse: 0.8973

l2
------- BaselineSVD
=== Part full
rmse: 0.9329
"""

# Get dataset
dataset = get_dataset('ml100k')

models = []

# # Hybrid Model
# from hybrid_model.hybrid import HybridModel
# from hybrid_model.config import hybrid_config
# model_type = HybridModel
# config = hybrid_config
# models.append(EvalModel(model_type.__name__, model_type, config))

# rmse: 0.9228

# # Bias Baseline
# from hybrid_model.baselines import BaselineBias
# model_type = BaselineBias
# config = dict(reg_bias=0.00005)
# models.append(EvalModel(model_type.__name__, model_type, config))

# # SVD
# from hybrid_model.baselines import BaselineSVD
# model_type = BaselineSVD
# config = dict(n_factors=40, reg_bias=0.00005, reg_latent=0.000001)
# models.append(EvalModel(model_type.__name__, model_type, config))

# SVD++
from hybrid_model.baselines import BaselineSVDpp
model_type = BaselineSVDpp
config = dict(n_factors=40, reg_bias=0.00004, reg_latent=0.00005, implicit_thresh=3.5)
models.append(EvalModel(model_type.__name__, model_type, config))

# from hybrid_model.baselines import AttributeBias
# model_type = AttributeBias
# config = dict(reg_att_bias=0.0015, reg_bias=0.0001)
# models.append(EvalModel(model_type.__name__, model_type, config))
#
# from hybrid_model.baselines import AttributeBiasExperimental
# model_type = AttributeBiasExperimental
# config = dict(reg_att_bias=0.0015, reg_bias=0.0001)
# models.append(EvalModel(model_type.__name__, model_type, config))

results = evaluate_models_single(dataset, models)

print_results(results)
