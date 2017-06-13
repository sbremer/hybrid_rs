import time

from evaluation.eval_script import evaluate_models_xval, print_results, EvalModel
from hybrid_model.dataset import get_dataset

dataset = get_dataset('ml100k')

models = []

# SVD++
from hybrid_model.baselines import BaselineSVDpp
model_type = BaselineSVDpp
config = dict(n_factors=40, reg_bias=0.00004, reg_latent=0.00005, implicit_thresh=0.0)
models.append(EvalModel(model_type.__name__, model_type, config))

start = time.time()
results = evaluate_models_xval(dataset, models)
end = time.time()

elapsed = end - start

print('Elapsed time: {}s'.format(elapsed))
print_results(results)

"""
rmsprop
Elapsed time: 107.84296679496765s
------- BaselineSVDpp
Combined Results:
=== Part full
rmse: 0.9074 ± 0.0015

Adagrad
Elapsed time: 111.42921781539917s
------- BaselineSVDpp
Combined Results:
=== Part full
rmse: 0.9060 ± 0.0008

Adadelta
Elapsed time: 611.3416509628296s
------- BaselineSVDpp
Combined Results:
=== Part full
rmse: 0.9069 ± 0.0008

Adam
Elapsed time: 96.75544929504395s
------- BaselineSVDpp
Combined Results:
=== Part full
rmse: 0.9103 ± 0.0019

adamax
Elapsed time: 125.60087895393372s
------- BaselineSVDpp
Combined Results:
=== Part full
rmse: 0.9067 ± 0.0018

nadam
Elapsed time: 97.68192219734192s
------- BaselineSVDpp
Combined Results:
=== Part full
rmse: 0.9091 ± 0.0021
"""
