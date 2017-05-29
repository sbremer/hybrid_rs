from eval_script import evaluate_models_single, evaluate_models_xval, print_results, EvalModel
from hybrid_model.dataset import get_dataset

"""
------- AttributeBias
Combined Results:
=== Part full
rmse: 1.0118 ± 0.0013

------- AttributeBiasExperimental
Combined Results:
=== Part full
rmse: 1.0114 ± 0.0008
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

# Bias Baseline
from hybrid_model.baselines import BaselineBias
model_type = BaselineBias
config = dict(reg_bias=0.00005)
models.append(EvalModel(model_type.__name__, model_type, config))

# # SVD
# from hybrid_model.baselines import BaselineSVD
# model_type = BaselineSVD
# config = dict(n_factors=40, reg_bias=0.00005, reg_latent=0.000001)
# models.append(EvalModel(model_type.__name__, model_type, config))

# # SVD++
# from hybrid_model.baselines import BaselineSVDpp
# model_type = BaselineSVDpp
# config = dict(n_factors=40, reg_bias=0.00004, reg_latent=0.00005, implicit_thresh=3.5)
# models.append(EvalModel(model_type.__name__, model_type, config))

# from hybrid_model.baselines import AttributeBias
# model_type = AttributeBias
# config = dict(reg_att_bias=0.0015, reg_bias=0.0001)
# models.append(EvalModel(model_type.__name__, model_type, config))
#
# from hybrid_model.baselines import AttributeBiasExperimental
# model_type = AttributeBiasExperimental
# config = dict(reg_att_bias=0.0015, reg_bias=0.0001)
# models.append(EvalModel(model_type.__name__, model_type, config))

results = evaluate_models_xval(dataset, models, user_coldstart=True)

print_results(results)
