import script_chdir
from evaluation.eval_script import evaluate_models_xval, EvalModel, print_results
from hybrid_model.dataset import get_dataset
import time


# Get dataset
dataset = get_dataset('ml100k')

models = []

# # Bias Baseline
# from hybrid_model.models import BiasEstimator
# model_type = BiasEstimator
# config = {}
# models.append(EvalModel(model_type.__name__, model_type, config))

# # AttributeBiasExperimental
# from hybrid_model.models import AttributeBiasAdvanced
# model_type = AttributeBiasAdvanced
# config = {}
# models.append(EvalModel(model_type.__name__, model_type, config))
#
# # AttributeLFF
# from hybrid_model.models import AttributeFactorization
# model_type = AttributeFactorization
# config = {}
# models.append(EvalModel(model_type.__name__, model_type, config))

# # SVDpp
# from hybrid_model.models import SVDpp
# model_type = SVDpp
# config = {}
# models.append(EvalModel(model_type.__name__, model_type, config))
#
# # SigmoidSVDpp
# from hybrid_model.models import SigmoidSVDpp
# model_type = SigmoidSVDpp
# config = {}
# models.append(EvalModel(model_type.__name__, model_type, config))

# # SigmoidUserAsymFactoring
# from hybrid_model.models import SigmoidUserAsymFactoring
# model_type = SigmoidUserAsymFactoring
# config = {'n_factors': 40, 'reg_bias': 0.000005, 'reg_latent': 0.00003,
#                      'implicit_thresh': 3.0, 'optimizer': 'adagrad'}
# models.append(EvalModel(model_type.__name__, model_type, config))

# # SigmoidItemAsymFactoring
# from hybrid_model.models import SigmoidItemAsymFactoring
# model_type = SigmoidItemAsymFactoring
# config = {}
# models.append(EvalModel(model_type.__name__, model_type, config))

# HybridModel
from hybrid_model.hybrid import HybridModel
from hybrid_model.config import hybrid_config_new
model_type = HybridModel
config = hybrid_config_new
models.append(EvalModel(model_type.__name__, model_type, config))

# # ExperimentalFactorization
# from hybrid_model.models.experimental import ExperimentalFactorization
# model_type = ExperimentalFactorization
# config = {'n_factors': 40, 'reg_bias': 0.000005, 'reg_latent': 0.00003,
#                      'implicit_thresh': 3.0, 'optimizer': 'adagrad'}
# models.append(EvalModel(model_type.__name__, model_type, config))

start = time.time()
results = evaluate_models_xval(dataset, models, coldstart=True, cs_type='user', n_entries=0)
end = time.time()

elapsed = end - start

print('Elapsed time: {}s'.format(elapsed))
print_results(results)

# Normal:
# Elapsed time: 691.0316336154938s
# ------- HybridModel
# RMSE: 0.8930 ± 0.0028  MAE: 0.7048 ± 0.0020  Prec@5: 0.7848 ± 0.0021  TopNAURC(k=100): 0.8987 ± 0.0015
# ------- HybridModel_SigmoidUserAsymFactoring
# RMSE: 0.8931 ± 0.0033  MAE: 0.7046 ± 0.0031  Prec@5: 0.7820 ± 0.0017  TopNAURC(k=100): 0.9041 ± 0.0010
# ------- HybridModel_AttributeBiasAdvanced
# RMSE: 0.9294 ± 0.0042  MAE: 0.7336 ± 0.0038  Prec@5: 0.7573 ± 0.0041  TopNAURC(k=100): 0.8200 ± 0.0022

# User
# Elapsed time: 415.66322684288025s
# ------- HybridModel
# RMSE: 1.0290 ± 0.0242  MAE: 0.8329 ± 0.0208  Prec@5: 0.5478 ± 0.0158  TopNAURC(k=100): 0.8198 ± 0.0105
# ------- HybridModel_SigmoidUserAsymFactoring
# RMSE: 1.2097 ± 0.0177  MAE: 0.9901 ± 0.0178  Prec@5: 0.4776 ± 0.0307  TopNAURC(k=100): 0.5668 ± 0.0093
# ------- HybridModel_AttributeBiasAdvanced
# RMSE: 1.0226 ± 0.0251  MAE: 0.8258 ± 0.0219  Prec@5: 0.5525 ± 0.0223  TopNAURC(k=100): 0.8107 ± 0.0113

# Item
# Elapsed time: 731.7550356388092s
# ------- HybridModel
# RMSE: 1.0603 ± 0.0146  MAE: 0.8675 ± 0.0141  Prec@5: 0.6595 ± 0.0221  TopNAURC(k=100): 0.7241 ± 0.0135
# ------- HybridModel_SigmoidUserAsymFactoring
# RMSE: 1.3043 ± 0.0222  MAE: 1.0898 ± 0.0225  Prec@5: 0.6035 ± 0.0240  TopNAURC(k=100): 0.4068 ± 0.0057
# ------- HybridModel_AttributeBiasAdvanced
# RMSE: 1.0477 ± 0.0137  MAE: 0.8563 ± 0.0132  Prec@5: 0.6651 ± 0.0216  TopNAURC(k=100): 0.6014 ± 0.0154
