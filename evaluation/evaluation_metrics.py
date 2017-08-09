import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt


class Metric:
    # Emtpy superclass

    def __str__(self):
        raise NotImplementedError


class BasicMetric(Metric):
    def calculate(self, y_true, y_pred, x) -> float:
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class Rmse(BasicMetric):
    def calculate(self, y_true, y_pred, x):
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        return rmse

    def __str__(self):
        return 'RMSE'


class Mae(BasicMetric):
    def calculate(self, y_true, y_pred, x):
        mae = mean_absolute_error(y_true, y_pred)
        return mae

    def __str__(self):
        return 'MAE'


class Precision(BasicMetric):
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

    def __str__(self):
        return 'Prec@{}'.format(self.k)


class Ndcg(BasicMetric):
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

    def __str__(self):
        return 'NDCG@{}'.format(self.k)


class AdvancedMetric(Metric):
    def calculate(self, model, x_train, x_test, y_test, y_pred) -> float:
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class TopNRecall(AdvancedMetric):
    """
    Method proposed by Cremonesi at RecSys2010
    """

    def __init__(self, k=100):
        # Number of random ratings to take
        self.k = k

    @staticmethod
    def _prep_user_data(model, u, x_train_u, x_train_i, x_test_u, x_test_i):

        # Get all item ratings for that user in the training set
        select_train_u = x_train_u == u
        items_train = x_train_i[select_train_u]

        # Get all item ratings for that user in the testing set
        select_test_u = x_test_u == u
        items_test = x_test_i[select_test_u]

        items = np.concatenate((items_train, items_test))

        # Get all items not yet rated by user (contained in the training and test set)
        items_test = np.setdiff1d(np.arange(model.n_items), items, assume_unique=True)
        user_test = np.full_like(items_test, u)

        # Predict ratings for not-rated items
        y = model.predict([user_test, items_test])

        return y

    def calculate(self, model, x_train, x_test, y_test, y_pred):

        # Filter to only use top ratings (== 5.0 for MovieLens
        top_y = y_test == np.max(y_test)

        # Select only top ratings from test set
        y_pred = y_pred[top_y]
        x_test_u = x_test[0][top_y].flatten()
        x_test_i = x_test[1][top_y].flatten()

        x_train_u = x_train[0]
        x_train_i = x_train[1]

        # Predicting all not rated items for all users in the test set might be slow and memory consuming!
        user_lookup = {}

        positions = []

        # Iterate over top testset ratings
        for y, u, i in zip(y_pred, x_test_u, x_test_i):

            # Predict ratings of non-rated items of not done for that user
            if u not in user_lookup:
                user_lookup[u] = self._prep_user_data(model, u, x_train_u, x_train_i, x_test_u, x_test_i)

            # Predictions for not-rated items by user u
            y_user = user_lookup[u]

            # Concat predicted (top) rating with k random (not trained) ratings of that user
            comparison = np.concatenate(([y], y_user[np.random.choice(len(y_user), self.k)]))

            # Find position of top rating
            pos = np.where(np.argsort(-comparison) == 0)[0][0]

            positions.append(pos)

        # Recall at fixed position
        # return np.sum(np.array(positions) < 10) / len(positions)

        # Area under Recall Curve
        return np.mean([np.sum(np.array(positions) < n) for n in range(1, self.k)]) / len(positions)

    def __str__(self):
        return 'TopNRecall(k={})'.format(self.k)
