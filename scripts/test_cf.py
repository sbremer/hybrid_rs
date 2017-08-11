import script_chdir
from evaluation.eval_script import evaluate_models_xval, EvalModel, print_results
from hybrid_model.dataset import get_dataset


# Get dataset
dataset = get_dataset('ml100k')

models = []

# Bias Baseline
from hybrid_model.models import BiasEstimator
model_type = BiasEstimator
config = {}
models.append(EvalModel(model_type.__name__, model_type, config))

# SVD
from hybrid_model.models import SVD
model_type = SVD
config = {}
models.append(EvalModel(model_type.__name__, model_type, config))

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

# SigmoidItemAsymFactoring
from hybrid_model.models import SigmoidItemAsymFactoring
model_type = SigmoidItemAsymFactoring
config = {'implicit_thresh': 4.0, 'n_factors': 79,
          'reg_bias': 0.004770353622067247, 'reg_latent': 2.3618479038250382e-05}
models.append(EvalModel(model_type.__name__, model_type, config))

# SigmoidUserAsymFactoring
from hybrid_model.models import SigmoidUserAsymFactoring
model_type = SigmoidUserAsymFactoring
config = {'implicit_thresh': 1.0, 'n_factors': 79,
          'reg_bias': 0.004770353622067247, 'reg_latent': 2.3618479038250382e-05}
models.append(EvalModel(model_type.__name__ + '_all_implicit', model_type, config))

# SigmoidUserAsymFactoring
from hybrid_model.models import SigmoidUserAsymFactoring
model_type = SigmoidUserAsymFactoring
config = {'implicit_thresh': 4.0, 'n_factors': 79,
          'reg_bias': 0.004770353622067247, 'reg_latent': 2.3618479038250382e-05}
models.append(EvalModel(model_type.__name__, model_type, config))


results = evaluate_models_xval(dataset, models, coldstart=False)

print_results(results)

# ------- BiasEstimator
# RMSE: 0.9419 ± 0.0050  MAE: 0.7457 ± 0.0041  Prec@5: 0.7485 ± 0.0027  TopNAURC(k=100): 0.8231 ± 0.0033  runtime: 9.3614 ± 2.2009
# ------- SVD
# RMSE: 0.9247 ± 0.0045  MAE: 0.7332 ± 0.0034  Prec@5: 0.7647 ± 0.0031  TopNAURC(k=100): 0.8505 ± 0.0018  runtime: 8.9738 ± 0.2339
# ------- SVDpp
# RMSE: 0.9006 ± 0.0036  MAE: 0.7130 ± 0.0039  Prec@5: 0.7790 ± 0.0032  TopNAURC(k=100): 0.8730 ± 0.0023  runtime: 15.5696 ± 0.5732
# ------- SigmoidSVDpp
# RMSE: 0.9003 ± 0.0028  MAE: 0.7120 ± 0.0030  Prec@5: 0.7780 ± 0.0032  TopNAURC(k=100): 0.8733 ± 0.0030  runtime: 15.6568 ± 0.4525
# ------- SigmoidItemAsymFactoring
# RMSE: 0.9292 ± 0.0040  MAE: 0.7407 ± 0.0043  Prec@5: 0.7777 ± 0.0033  TopNAURC(k=100): 0.8822 ± 0.0023  runtime: 18.6283 ± 0.7879
# ------- SigmoidUserAsymFactoring_all_implicit
# RMSE: 0.8928 ± 0.0030  MAE: 0.7034 ± 0.0018  Prec@5: 0.7793 ± 0.0020  TopNAURC(k=100): 0.8906 ± 0.0027  runtime: 17.0272 ± 0.7476
# ------- SigmoidUserAsymFactoring
# RMSE: 0.8931 ± 0.0032  MAE: 0.7040 ± 0.0033  Prec@5: 0.7825 ± 0.0022  TopNAURC(k=100): 0.9042 ± 0.0024  runtime: 16.0776 ± 0.6156
