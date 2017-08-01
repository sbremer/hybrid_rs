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

results = evaluate_models_xval(dataset, models, coldstart=True, cs_type='user', n_entries=0)

print_results(results)

# 1: 0.8903
# 2: 0.8872
# 3: 0.8859
# 4: 0.8875
# 5: 0.8978

# ------- HybridModel
# === Part full
# rmse: 0.8852 ± 0.0029  mae: 0.6965 ± 0.0025  prec@5: 0.7832 ± 0.0024
#
# ------- HybridModel_SigmoidUserAsymFactoring
# === Part full
# rmse: 0.8864 ± 0.0034  mae: 0.6969 ± 0.0030  prec@5: 0.7808 ± 0.0026
#
# ------- HybridModel_AttributeBiasAdvanced
# === Part full
# rmse: 0.9265 ± 0.0038  mae: 0.7331 ± 0.0035  prec@5: 0.7589 ± 0.0017