import numpy as np

np.random.seed(0)

from evaluation.evaluation import Evaluation, EvaluationResults, EvaluationResultsHybrid
from hybrid_model.hybrid import HybridModel
from hybrid_model.config import hybrid_config
from hybrid_model.dataset import get_dataset
from util import kfold

user_coldstart = False
n_entries = 0
n_fold = 5

epochs = 10

evaluation = Evaluation()
dataset = get_dataset('ml100k')

results_models = [EvaluationResultsHybrid() for _ in range(epochs + 1)]
results_hybrid = [EvaluationResults() for _ in range(epochs + 1)]

(inds_u, inds_i, y, users_features, items_features) = dataset.data

if user_coldstart:
    if n_entries == 0:
        fold = kfold.kfold_entries(n_fold, inds_u)
    else:
        fold = kfold.kfold_entries_plus(n_fold, inds_u, n_entries)
else:
    fold = kfold.kfold(n_fold, inds_u)

fold = list(fold)

for xval_train, xval_test in fold:

    # Dataset training
    inds_u_train = inds_u[xval_train]
    inds_i_train = inds_i[xval_train]
    y_train = y[xval_train]
    n_train = len(y_train)

    # Dataset testing
    inds_u_test = inds_u[xval_test]
    inds_i_test = inds_i[xval_test]
    y_test = y[xval_test]

    train = ([inds_u_train, inds_i_train], y_train)
    test = ([inds_u_test, inds_i_test], y_test)

    hybrid_model = HybridModel(users_features, items_features, hybrid_config, verbose=0)

    hybrid_model.fit_init(*train)

    result_model = evaluation.evaluate_hybrid(hybrid_model, *test)
    results_models[0].add(result_model)

    result_hybrid = evaluation.evaluate(hybrid_model, *test)
    results_hybrid[0].add(result_hybrid)

    for e in range(epochs):
        hybrid_model.fit_cross_epoch()

        result = evaluation.evaluate_hybrid(hybrid_model, *test)
        results_models[e + 1].add(result)

        result_hybrid = evaluation.evaluate(hybrid_model, *test)
        results_hybrid[e + 1].add(result_hybrid)

rmses_cf = []
rmses_md = []
rmses_hybrid = []

for e in range(epochs + 1):
    rmses_cf.append(results_models[e].mean_rmse_cf())
    rmses_md.append(results_models[e].mean_rmse_md())
    rmses_hybrid.append(results_hybrid[e].rmse())

    # print('Results after epoch {}:'.format(e))
    # print(results_models[e])

print('rmses_epochs_cf =', rmses_cf)
print('rmses_epochs_md =', rmses_md)
print('rmses_epochs_hybrid =', rmses_hybrid)

"""
# Normal
rmses_epochs_cf = [0.89960148840391463, 0.89779517729386416, 0.89672559940305019, 0.89626290401741215, 0.89578679881217804, 0.89561542943160377, 0.89567319920943567, 0.89551506213611276, 0.89570706473919093, 0.89575679537764541, 0.89575830777449195]
rmses_epochs_md = [0.9265575851917649, 0.92583061093396068, 0.92528327829188939, 0.92495449742979619, 0.92465154436997887, 0.92431301872002847, 0.92413590792169642, 0.92391674979954352, 0.92369720645138498, 0.92350945550720509, 0.92339056888259974]
rmses_epochs_hybrid = [0.89966098335769318, 0.89755639939507892, 0.8965226571710122, 0.89607192989109608, 0.89560595280242195, 0.89550923249699732, 0.89553063070340022, 0.89539816499084035, 0.89561169299303722, 0.89561603677956969, 0.89564290242563105]

"""