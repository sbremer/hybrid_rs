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

# SVDpp
from hybrid_model.models import SVDpp
model_type = SVDpp
config = {}
models.append(EvalModel(model_type.__name__, model_type, config))

# SigmoidSVDpp
from hybrid_model.models import SigmoidSVDpp
model_type = SigmoidSVDpp
config = {}
models.append(EvalModel(model_type.__name__, model_type, config))

results = evaluate_models_xval(dataset, models, coldstart=False, cs_type='user', n_entries=0)

print_results(results)
