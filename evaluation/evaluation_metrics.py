import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt


class Metric:
    def calculate(self, y_true, y_pred, x) -> float:
        raise NotImplementedError


class Rmse(Metric):
    def calculate(self, y_true, y_pred, x):
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        return rmse


class Mae(Metric):
    def calculate(self, y_true, y_pred, x):
        mae = mean_absolute_error(y_true, y_pred)
        return mae


class Precision(Metric):
    def __init__(self, k):
        self.k = k

    def calculate(self, y_true, y_pred, x):
        inds_u = x[0]
        users = np.unique(inds_u)

        precisions = np.zeros((len(users),))

        for i, user in enumerate(users):
            user_i = (inds_u == user).flatten()

            y_true_u = y_true[user_i]
            y_pred_u = y_pred[user_i]

            sort_true = np.argsort(-y_true_u)
            sort_pred = np.argsort(-y_pred_u)

            pred = list(sort_pred[:self.k])
            true = list(sort_true[:self.k])

            for r in sort_true[self.k:]:
                if y_true_u[r] == y_true_u[true[-1]]:
                    true.append(r)
                else:
                    break

            precision_u = sum(1 for p in pred if p in true) / min(self.k, len(pred))
            precisions[i] = precision_u

        return np.mean(precisions)


class Ndcg(Metric):
    def __init__(self, k):
        self.k = k

    def calculate(self, y_true, y_pred, x):
        inds_u = x[0]
        users, counts = np.unique(inds_u, return_counts=True)

        idcg = np.sum(np.ones((self.k,)) / np.log2(np.arange(2, self.k + 2)))

        dcgs = []

        for i, user in enumerate(users):
            user_i = (inds_u == user).flatten()

            y_true_u = y_true[user_i]
            y_pred_u = y_pred[user_i]

            sort_true = np.argsort(-y_true_u)
            sort_pred = np.argsort(-y_pred_u)

            pred = list(sort_pred[:self.k])
            true = list(sort_true[:self.k])

            for r in sort_true[self.k:]:
                if y_true_u[r] == y_true_u[true[-1]]:
                    true.append(r)
                else:
                    break

            r = [int(a in sort_true[:self.k]) for a in sort_pred[:self.k]]

            dcg = np.sum(r / np.log2(np.arange(2, self.k + 2))) / idcg
            dcgs.append(dcg)

        return np.mean(dcgs)
