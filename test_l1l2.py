from eval_script import evaluate_models_single, print_models_single, EvalModel
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
"""

# Get dataset
dataset = get_dataset('ml100k')

models = []

# from hybrid_model.baselines import BaselineBias
# model = BaselineBias(n_users, n_items)
# models.append((model.__class__.__name__, model))

# # Hybrid Model
# from hybrid_model.hybrid import HybridModel
# from hybrid_model.config import hybrid_config
# model_type = HybridModel
# config = hybrid_config
# models.append(EvalModel(model_type.__name__, model_type, config))

# SVD++
from hybrid_model.baselines import BaselineSVDpp
model_type = BaselineSVDpp
config = dict(n_factors=40, reg_bias=0.00004, reg_latent=0.00005, implicit_thresh=3.5)
models.append(EvalModel(model_type.__name__, model_type, config))

# from hybrid_model.baselines import AttributeBias
# model = AttributeBias(users_features, items_features)
# models.append((model.__class__.__name__, model))

# from hybrid_model.baselines import AttributeBiasExperimental
# model = AttributeBiasExperimental(users_features, items_features)
# models.append((model.__class__.__name__, model))

results = evaluate_models_single(dataset, models)

print_models_single(results)
