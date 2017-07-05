import script_chdir
import time
from evaluation.eval_script import evaluate_models_xval, print_results, EvalModel
from hybrid_model.dataset import get_dataset

dataset = get_dataset('ml100k')

models = []

# Hybrid Model
from hybrid_model.hybrid import HybridModel
from hybrid_model.config import hybrid_config

model_type = HybridModel
config = hybrid_config
models.append(EvalModel(model_type.__name__, model_type, config))

start = time.time()
results = evaluate_models_xval(dataset, models)
end = time.time()

elapsed = end - start

print('Elapsed time: {}s'.format(elapsed))
print_results(results)

"""
Elapsed time: 317.58788084983826s
------- HybridModel
=== Part full
rmse: 0.8959 ± 0.0025  prec@5: 0.7820 ± 0.0032

------- HybridModel_SVDpp
=== Part full
rmse: 0.9002 ± 0.0028  prec@5: 0.7774 ± 0.0021

------- HybridModel_AttributeBiasExperimental
=== Part full
rmse: 0.9266 ± 0.0038  prec@5: 0.7593 ± 0.0027
"""
