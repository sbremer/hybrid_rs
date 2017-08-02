import script_chdir
from evaluation.eval_script import evaluate_models_xval, EvalModel, print_results
from hybrid_model.dataset import get_dataset


# Get dataset
dataset = get_dataset('ml100k')

models = []

# # Bias Baseline
# from hybrid_model.models import BiasEstimator
# model_type = BiasEstimator
# config = {}
# models.append(EvalModel(model_type.__name__, model_type, config))

# # AttributeBiasExperimental
# from hybrid_model.models import AttributeBiasExperimental
# model_type = AttributeBiasExperimental
# config = {}
# models.append(EvalModel(model_type.__name__, model_type, config))
#
# # AttributeLFF
# from hybrid_model.models import AttributeLFF
# model_type = AttributeLFF
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

results = evaluate_models_xval(dataset, models, coldstart=False, cs_type='user', n_entries=0)

print_results(results)

# ------- HybridModel
# RMSE: 0.8961 ± 0.0032  MAE: 0.7071 ± 0.0035  Prec@5: 0.7813 ± 0.0016  TopNRecall(k=100): 0.8468 ± 0.0030
# ------- HybridModel_SVDpp
# RMSE: 0.9001 ± 0.0035  MAE: 0.7116 ± 0.0037  Prec@5: 0.7781 ± 0.0036  TopNRecall(k=100): 0.8695 ± 0.0029
# ------- HybridModel_AttributeBiasAdvanced
# RMSE: 0.9269 ± 0.0047  MAE: 0.7336 ± 0.0041  Prec@5: 0.7572 ± 0.0015  TopNRecall(k=100): 0.8177 ± 0.0016
# ------- BiasEstimator
# RMSE: 0.9420 ± 0.0050  MAE: 0.7458 ± 0.0041  Prec@5: 0.7485 ± 0.0026  TopNRecall(k=100): 0.8210 ± 0.0029
# ------- SVD
# RMSE: 0.9248 ± 0.0040  MAE: 0.7331 ± 0.0030  Prec@5: 0.7652 ± 0.0021  TopNRecall(k=100): 0.8481 ± 0.0023

# ------- HybridModel
# RMSE: 0.8827 ± 0.0025  MAE: 0.6952 ± 0.0021  Prec@5: 0.7872 ± 0.0022  TopNRecall(k=100): 0.8614 ± 0.0011
# ------- HybridModel_SigmoidUserAsymFactoring
# RMSE: 0.8847 ± 0.0030  MAE: 0.6962 ± 0.0023  Prec@5: 0.7824 ± 0.0009  TopNRecall(k=100): 0.8883 ± 0.0019
# ------- HybridModel_AttributeBiasAdvanced
# RMSE: 0.9266 ± 0.0048  MAE: 0.7326 ± 0.0041  Prec@5: 0.7580 ± 0.0028  TopNRecall(k=100): 0.8181 ± 0.0021