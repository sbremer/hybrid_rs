import numpy as np

np.random.seed(0)

from evaluation.evaluation import Evaluation, EvaluationResultsHybrid
from hybrid_model.hybrid import HybridModel
from hybrid_model.config import hybrid_config as hybrid_config
from hybrid_model.dataset import get_dataset
from util import kfold

user_coldstart = False
n_entries = 0
n_fold = 5

epochs = 10

evaluation = Evaluation()
dataset = get_dataset('ml100k')

results = [EvaluationResultsHybrid() for _ in range(epochs + 1)]

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
    result = evaluation.evaluate_hybrid(hybrid_model, *test)

    results[0].add(result)

    hybrid_model.setup_cross_training()

    for e in range(epochs):
        hybrid_model.fit_cross_epoch()
        result = evaluation.evaluate_hybrid(hybrid_model, *test)

        results[e + 1].add(result)

rmses_cf = []
rmses_md = []

for e in range(epochs + 1):
    rmses_cf.append(results[e].mean_rmse_cf())
    rmses_md.append(results[e].mean_rmse_md())

    print('Results after epoch {}:'.format(e))
    print(results[e])

print('rmses_epochs_cf =', rmses_cf)
print('rmses_epochs_md =', rmses_md)

"""
Normal:
rmses_epochs_cf = [0.89871391966622272, 0.89801953407399149, 0.89637120263350345, 0.89622641955300364, 0.89645749534175057, 0.89679681842880066, 0.8970646800071076, 0.89739404569727077, 0.897932696039214, 0.89840521783122473, 0.89894268211972972]
rmses_epochs_md = [0.92768680982736773, 0.92508630348373833, 0.92395628519279782, 0.92358965970879559, 0.923284393281261, 0.92305838341505297, 0.92300587014004787, 0.92291819317069412, 0.92284853432111957, 0.92284402245768715, 0.92288891004958629]
Coldstart:
rmses_epochs_cf = [1.0890348272091348, 1.0227211255627047, 1.0217580107038859, 1.0213056047008608, 1.0211928858521013, 1.0210529638365688, 1.0207690127529934, 1.0205373643498017, 1.0207205203293839, 1.020805172320383, 1.0202658436606722]
rmses_epochs_md = [1.019422290642984, 1.0178845054797354, 1.0174474322784191, 1.0172457871871889, 1.0169962388874103, 1.0168267183304573, 1.0169104186621039, 1.0169892430200405, 1.0171037238296456, 1.0169569270858083, 1.0170247108552808]

"""