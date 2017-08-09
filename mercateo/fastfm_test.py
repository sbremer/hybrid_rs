import script_chdir
import numpy as np
np.random.seed(0)

from fastFM.mcmc import FMRegression
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

from hybrid_model import dataset
from evaluation import evaluation
from util import kfold

ds = dataset.get_dataset('ml100k')

(inds_u, inds_i, y, users_features, items_features) = ds.data

X_features = sparse.csr_matrix(np.concatenate((users_features[inds_u], items_features[inds_i]), axis=1))
X_ids = np.concatenate((inds_u[:, None], inds_i[:, None]), axis=1)
# X_ids = inds_i[:, None]

encoder = OneHotEncoder(handle_unknown='ignore').fit(X_ids)
X_ids = encoder.transform(X_ids).tocsr()

implicit = sparse.coo_matrix((np.ones((len(inds_u),)), (inds_u, inds_i))).tocsr()

# X = sparse.hstack((X_ids, X_features, implicit[inds_u, :])).tocsr()
X = sparse.hstack((X_ids, X_features)).tocsr()
# X = X_features

n_fold = 5

folds = list(kfold.kfold(n_fold, inds_u))
# folds = list(kfold.kfold_entries(n_fold, inds_u))
# folds = list(kfold.kfold_entries(n_fold, inds_i))

# evaluater = evaluation.Evaluation()
results = evaluation.EvaluationResultsPart(evaluation.metrics_rmse_prec.keys())

for xval_train, xval_test in folds:

    X_train = X[xval_train, :]
    X_test = X[xval_test, :]

    y_train = y[xval_train]
    y_test = y[xval_test]

    inds_u_test = inds_u[xval_test]
    inds_i_test = inds_i[xval_test]

    model = FMRegression()
    # model = FMRegression(n_iter=1000, init_stdev=0.1, rank=40, random_state=123, l2_reg_w=0.01, l2_reg_V=0.01, l2_reg=0, step_size=0.05)

    y_pred = model.fit_predict(X_train, y_train, X_test)
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)

    result = evaluation.EvaluationResultPart()
    for measure, metric in evaluation.metrics_rmse_prec.items():
        result.results[measure] = metric.calculate(y_test, y_pred, [inds_u_test, inds_i_test])

    print(result)
    results.add(result)

print(results)


