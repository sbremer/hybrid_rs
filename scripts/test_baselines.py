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

# AttributeBiasExperimental
from hybrid_model.models import AttributeBiasExperimental
model_type = AttributeBiasExperimental
config = {}
models.append(EvalModel(model_type.__name__, model_type, config))

# AttributeLFF
from hybrid_model.models import AttributeLFF
model_type = AttributeLFF
config = {}
models.append(EvalModel(model_type.__name__, model_type, config))

results = evaluate_models_xval(dataset, models, coldstart=True, cs_type='item', n_entries=0)

print_results(results)
