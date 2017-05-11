import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# Local
# import hybrid_model.rank_metrics as rm


def rmse(y_true, y_pred):
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return rmse


def mae(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    return mae


def ndcg(y_true, y_pred, k, inds_u):
    users, counts = np.unique(inds_u, return_counts=True)

    idcg = np.sum(np.ones((k,)) / np.log2(np.arange(2, k + 2)))

    dcgs = []

    for i, user in enumerate(users):
        user_i = (inds_u == user).flatten()

        if np.sum(user_i) <= k:
            continue

        y_true_u = y_true[user_i]
        y_pred_u = y_pred[user_i]

        sort_true = np.argsort(-y_true_u)
        sort_pred = np.argsort(-y_pred_u)

        r = [int(a in sort_true[:k]) for a in sort_pred[:k]]

        dcg = np.sum(r / np.log2(np.arange(2, k + 2))) / idcg
        dcgs.append(dcg)

    return np.mean(dcgs)

