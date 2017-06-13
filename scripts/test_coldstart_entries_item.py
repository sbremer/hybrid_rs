from evaluation.eval_script import evaluate_models_xval, EvalModel
from hybrid_model.dataset import get_dataset

# Get dataset
dataset = get_dataset('ml100k')
# dataset = get_dataset('ml1m')

models = []

# Hybrid Model
from hybrid_model.hybrid import HybridModel
from hybrid_model.config import hybrid_config

model_type = HybridModel
config = hybrid_config
models.append(EvalModel(model_type.__name__, model_type, config))

params_entries = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
results = {}

rmses_before_cf = []
rmses_before_md = []
rmses_after_cf = []
rmses_after_md = []

for entries_plus in params_entries:
    result = evaluate_models_xval(dataset, models, coldstart=True, cs_type='item', n_entries=entries_plus, repeat=1)
    results[entries_plus] = result

    rmses_before_cf.append(result[0][1][0].mean_rmse_cf())
    rmses_before_md.append(result[0][1][0].mean_rmse_md())

    rmses_after_cf.append(result[0][1][1].mean_rmse_cf())
    rmses_after_md.append(result[0][1][1].mean_rmse_md())

print('rmses_before_cf =', rmses_before_cf)
print('rmses_before_md =', rmses_before_md)
print('rmses_after_cf =', rmses_after_cf)
print('rmses_after_md =', rmses_after_md)

"""
rmses_before_cf = [1.1317312347267516, 0.98360425047828348, 0.94713060429363805, 0.93540847769920155, 0.92150267872241298, 0.91507544138497787, 0.90925541551548505, 0.90550861417222461, 0.90572022185833601, 0.89665085291617141, 0.89988160743265877, 0.8902564356368885, 0.89242298431801037]
rmses_before_md = [1.0616732547167733, 0.98697412978429333, 0.96753236906916629, 0.95993779196221296, 0.93559763846230459, 0.93318960697509801, 0.93568229901975997, 0.92755503335869105, 0.92918612735146211, 0.92297412884290575, 0.93031717221823595, 0.92071621862027231, 0.92054645230410015]
"""
