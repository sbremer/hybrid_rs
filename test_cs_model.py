from eval_script import evaluate_models_single, evaluate_models_xval, print_results, EvalModel
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

# rmse: 1.0107 ± 0.0002

# Bias Baseline
from hybrid_model.baselines import BaselineBias
model_type = BaselineBias
config = dict(reg_bias=0.000003)
models.append(EvalModel(model_type.__name__, model_type, config))

# # SVD
# from hybrid_model.baselines import BaselineSVD
# model_type = BaselineSVD
# config = dict(n_factors=40, reg_bias=0.000004, reg_latent=0.000001)
# models.append(EvalModel(model_type.__name__, model_type, config))

# # SVD++
# from hybrid_model.baselines import BaselineSVDpp
# model_type = BaselineSVDpp
# config = dict(n_factors=40, reg_bias=0.00004, reg_latent=0.00005, implicit_thresh=3.5)
# models.append(EvalModel(model_type.__name__, model_type, config))

# from hybrid_model.baselines import AttributeBias
# model_type = AttributeBias
# config = dict(reg_att_bias=0.0015, reg_bias=0.00005)
# models.append(EvalModel(model_type.__name__, model_type, config))

# from hybrid_model.baselines import AttributeBiasExperimental
# model_type = AttributeBiasExperimental
# config = dict(reg_att_bias=0.00001, reg_bias=0.0000001)
# models.append(EvalModel(model_type.__name__, model_type, config))

results = evaluate_models_xval(dataset, models, user_coldstart=True, n_entries=0)
print('Coldstart')
print_results(results)

"""
ml1m
0: Bias   AttE 0.9785 SVD++ 0.9947 SVD
20: AttE 0.9113 SVD++ 0.9420
30: AttE  CF
"""

results = evaluate_models_xval(dataset, models, user_coldstart=False)
print('Normal')
print_results(results)