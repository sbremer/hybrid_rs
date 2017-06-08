import matplotlib.pyplot as plt
import numpy as np

from evaluation.evaluation import EvaluationResults
from evaluation.evaluation import n_bins

results_before_xtrain: EvaluationResults
results_after_xtrain: EvaluationResults

user_bins_before_mf = np.zeros((n_bins,))
user_bins_after_mf = np.zeros((n_bins,))

item_bins_before_mf = np.zeros((n_bins,))
item_bins_after_mf = np.zeros((n_bins,))

for i in range(n_bins):
    bin_name = 'user_{}'.format(i+1)

    results_part = results_before_xtrain.parts[bin_name]
    result = results_part.model_mf.results['rmse']
    user_bins_before_mf[i] = np.mean(result)

    results_part = results_after_xtrain.parts[bin_name]
    result = results_part.model_mf.results['rmse']
    user_bins_after_mf[i] = np.mean(result)

    bin_name = 'item_{}'.format(i + 1)

    results_part = results_before_xtrain.parts[bin_name]
    result = results_part.model_mf.results['rmse']
    item_bins_before_mf[i] = np.mean(result)

    results_part = results_after_xtrain.parts[bin_name]
    result = results_part.model_mf.results['rmse']
    item_bins_after_mf[i] = np.mean(result)

x = np.arange(n_bins)
plt.plot(x, user_bins_before_mf, 'r--', x, user_bins_after_mf, 'r-')
plt.plot(x, item_bins_before_mf, 'b--', x, item_bins_after_mf, 'b-')
plt.show()

# CS

user_bins_before_cs = np.zeros((n_bins,))
user_bins_after_cs = np.zeros((n_bins,))

item_bins_before_cs = np.zeros((n_bins,))
item_bins_after_cs = np.zeros((n_bins,))

for i in range(n_bins):
    bin_name = 'user_{}'.format(i+1)

    results_part = results_before_xtrain.parts[bin_name]
    result = results_part.model_cs.results['rmse']
    user_bins_before_cs[i] = np.mean(result)

    results_part = results_after_xtrain.parts[bin_name]
    result = results_part.model_cs.results['rmse']
    user_bins_after_cs[i] = np.mean(result)

    bin_name = 'item_{}'.format(i + 1)

    results_part = results_before_xtrain.parts[bin_name]
    result = results_part.model_cs.results['rmse']
    item_bins_before_cs[i] = np.mean(result)

    results_part = results_after_xtrain.parts[bin_name]
    result = results_part.model_cs.results['rmse']
    item_bins_after_cs[i] = np.mean(result)

x = np.arange(n_bins)
plt.plot(x, user_bins_before_cs, 'r--', x, user_bins_after_cs, 'r-')
plt.plot(x, item_bins_before_cs, 'b--', x, item_bins_after_cs, 'b-')
plt.show()


# Combined Plot

x = np.arange(n_bins)
plt.plot(x, user_bins_before_cs, 'r--', x, user_bins_before_mf, 'r-')
plt.plot(x, item_bins_before_cs, 'b--', x, item_bins_before_mf, 'b-')
plt.show()