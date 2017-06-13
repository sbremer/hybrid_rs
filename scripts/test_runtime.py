import scripts
import time
from evaluation.eval_script import evaluate_models_xval, print_results, EvalModel
from hybrid_model.dataset import get_dataset

dataset = get_dataset('ml100k')

models = []

# SVD++
from hybrid_model.models import SVDpp
model_type = SVDpp
config = {}
models.append(EvalModel(model_type.__name__, model_type, config))

start = time.time()
results = evaluate_models_xval(dataset, models)
end = time.time()

elapsed = end - start

print('Elapsed time: {}s'.format(elapsed))
print_results(results)

"""
Elapsed time: 93.90857577323914s
------- SVDpp
=== Part full
rmse: 0.9004 ± 0.0034  prec@5: 0.7789 ± 0.0037
"""
