import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt

# Global stuff
n_users, n_items = 0, 0


# <editor-fold desc="Auxiliary Functions">
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))
# </editor-fold>


# <editor-fold desc="Recommender Functions">
def cf_bias_matrix(train_data_matrix, test_data_matrix):
    import numpy.ma as ma
    train_data_matrix = ma.masked_values(train_data_matrix, 0)

    ratings_total = train_data_matrix.mean()
    ratings_user = np.array(train_data_matrix.mean(axis=1) - ratings_total)
    ratings_item = np.array(train_data_matrix.mean(axis=0) - ratings_total)

    predicted_g = np.tile(ratings_total, (n_users, n_items))
    predicted_gui = ratings_total + np.tile(ratings_user, (n_items, 1)).T + np.tile(ratings_item, (n_users, 1))

    rmse_g = rmse(predicted_g, test_data_matrix)
    rmse_gui = rmse(predicted_gui, test_data_matrix)

    # print('RMSE using only global bias: {}'.format(rmse_g))
    # print('RMSE using global, user and item bias: {}'.format(rmse_gui))

    return rmse_g, rmse_gui


# Obsolete
def cf_bias(train, test):

    ratings_user = np.zeros(n_users)
    ratings_user_n = np.zeros(n_users)
    ratings_item = np.zeros(n_items)
    ratings_item_n = np.zeros(n_items)

    ratings_total = 0

    A = np.zeros((n_users, n_items))
    for line in train.itertuples():
        user = line[1] - 1
        item = line[2] - 1
        rating = line[3]

        A[user, item] = rating
        ratings_user[user] += rating
        ratings_user_n[user] += 1
        ratings_item[item] += rating
        ratings_item_n[item] += 1
        ratings_total += rating

    # Workaround for zero divison
    ratings_item_n[ratings_item_n == 0] = 1
    ratings_user_n[ratings_user_n == 0] = 1

    # Global average
    ratings_total /= len(train)

    # User average (difference from global average)
    ratings_user = (ratings_user / ratings_user_n) - ratings_total

    # Item average (difference from global average)
    ratings_item = (ratings_item / ratings_item_n) - ratings_total

    rmse_g = 0
    rmse_gui = 0

    for line in test.itertuples():
        user = line[1] - 1
        item = line[2] - 1
        rating = line[3]

        rmse_g += (rating - ratings_total) ** 2

        rating_predict = ratings_total + ratings_user[user] + ratings_item[item]
        rmse_gui += (rating - rating_predict) ** 2

    rmse_g /= len(test)
    rmse_gui /= len(test)

    # print('RMSE using only global bias: {}'.format(rmse_g))
    # print('RMSE using global, user and item bias: {}'.format(rmse_gui))

    return rmse_g, rmse_gui


def cf_similarity(train_data_matrix, test_data_matrix):

    from sklearn.metrics.pairwise import pairwise_distances
    user_similarity = pairwise_distances(train_data_matrix, metric='euclidean')
    item_similarity = pairwise_distances(train_data_matrix.T, metric='euclidean')

    def predict(ratings, similarity, type='user'):
        if type == 'user':
            mean_user_rating = ratings.mean(axis=1)
            # You use np.newaxis so that mean_user_rating has same format as ratings
            ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
            pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array(
                [np.abs(similarity).sum(axis=1)]).T
        elif type == 'item':
            pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
        return pred

    item_prediction = predict(train_data_matrix, item_similarity, type='item')
    user_prediction = predict(train_data_matrix, user_similarity, type='user')

    rmse_user = rmse(user_prediction, test_data_matrix)
    rmse_item = rmse(item_prediction, test_data_matrix)

    # print('User-based CF RMSE: ' + str(rmse_user))
    # print('Item-based CF RMSE: ' + str(rmse_item))

    return rmse_user, rmse_item


def cf_svd(train_data_matrix, test_data_matrix):
    from scipy.sparse.linalg import svds

    u, s, vt = svds(train_data_matrix, k=40)
    s_diag_matrix = np.diag(s)
    X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
    rmse_svd = rmse(X_pred, test_data_matrix)

    # print('SVD CF MSE: ' + str(rmse_svd))
    return rmse_svd


def cf_mf(train_data_matrix, test_data_matrix):
    import implicit
    from scipy.sparse import coo_matrix

    train_data_matrix = train_data_matrix.T
    indices = np.nonzero(train_data_matrix)
    train_data_matrix = (train_data_matrix - 3) / 2
    cm = coo_matrix((train_data_matrix[indices], indices), shape=train_data_matrix.shape)

    rank = 100
    model = implicit.als.AlternatingLeastSquares(factors=rank, num_threads=8, regularization=0.1, calculate_training_loss=True, iterations=15)
    # model.user_factors = np.random.rand(n_users, rank).astype(np.float64) * 0.01
    # model.item_factors = np.random.rand(n_items, rank).astype(np.float64) * 0.01
    model.fit(cm)

    # user_factors, item_factors = implicit.alternating_least_squares(cm, factors=500)
    predicted = model.user_factors.dot(model.item_factors.T) * 2 + 3

    rmse_mf = rmse(predicted, test_data_matrix)

    print('MF CF MSE: ' + str(rmse_mf))
    return rmse_mf


def cf_mf2(train_data_matrix, test_data_matrix):
    lambda_ = 0.1
    n_factors = 100
    m, n = train_data_matrix.shape
    n_iterations = 20

    X = 5 * np.random.rand(m, n_factors)
    Y = 5 * np.random.rand(n_factors, n)

    def get_error(Q, X, Y, W):
        return np.sum((W * (Q - np.dot(X, Y))) ** 2)

    W = (train_data_matrix != 0).astype(np.float64, copy=False)

    weighted_errors = []
    for ii in range(n_iterations):
        for u, Wu in enumerate(W):
            X[u] = np.linalg.solve(np.dot(Y, np.dot(np.diag(Wu), Y.T)) + lambda_ * np.eye(n_factors),
                                   np.dot(Y, np.dot(np.diag(Wu), train_data_matrix[u].T))).T
        for i, Wi in enumerate(W.T):
            Y[:, i] = np.linalg.solve(np.dot(X.T, np.dot(np.diag(Wi), X)) + lambda_ * np.eye(n_factors),
                                      np.dot(X.T, np.dot(np.diag(Wi), train_data_matrix[:, i])))
        weighted_errors.append(get_error(train_data_matrix, X, Y, W))
        print('{}th iteration is completed'.format(ii))
    weighted_Q_hat = np.dot(X, Y)
    print('Error of rated movies: {}'.format(get_error(train_data_matrix, X, Y, W)))

    pass

# </editor-fold>


def main():
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv('data/ml-100k/u.data', sep='\t', names=header)

    global n_users, n_items
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]
    print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

    n_fold = 5
    kf = KFold(n_splits=n_fold, shuffle=True)

    # Config of what to run (Similarity takes long)
    run_bias = False
    run_sim = False
    run_svd = False
    run_mf = True

    rmses_bias_g = []
    rmses_bias_gui = []
    rmses_sim_user = []
    rmses_sim_item = []
    rmses_svd = []
    rmses_mf = []

    for train_indices, test_indices in kf.split(df):
        train = df.iloc[train_indices, :]
        test = df.iloc[test_indices, :]

        train_data_matrix = np.zeros((n_users, n_items))
        for line in train.itertuples():
            train_data_matrix[line[1] - 1, line[2] - 1] = line[3]

        test_data_matrix = np.zeros((n_users, n_items))
        for line in test.itertuples():
            test_data_matrix[line[1] - 1, line[2] - 1] = line[3]

        # Bias CF
        if run_bias:
            rmse_bias_g, rmse_bias_gui = cf_bias_matrix(train_data_matrix, test_data_matrix)

            rmses_bias_g.append(rmse_bias_g)
            rmses_bias_gui.append(rmse_bias_gui)

        # Similarity CF
        if run_sim:
            rmse_sim_user, rmse_sim_item = cf_similarity(train_data_matrix, test_data_matrix)

            rmses_sim_user.append(rmse_sim_user)
            rmses_sim_item.append(rmse_sim_item)

        # SVD CF
        if run_svd:
            rmse_svd = cf_svd(train_data_matrix, test_data_matrix)

            rmses_svd.append(rmse_svd)

        # MF CF
        if run_mf:
            rmse_mf = cf_mf2(train_data_matrix, test_data_matrix)

            rmses_mf.append(rmse_mf)


    # Bias CF
    if run_bias:
        print('Crossval RMSE of global bias: {}'.format(np.mean(rmses_bias_g)))
        print('Crossval RMSE of global, user and item bias: {}'.format(np.mean(rmses_bias_gui)))

    # Similarity CF
    if run_sim:
        print('Crossval RMSE of User-similarity-based CF: {}'.format(np.mean(rmses_sim_user)))
        print('Crossval RMSE of Item-similarity-based CF: {}'.format(np.mean(rmses_sim_item)))

    # SVD CF
    if run_svd:
        print('Crossval RMSE of SVD-based CF: {}'.format(np.mean(rmses_svd)))

    # MF CF
    if run_mf:
        print('Crossval RMSE of MF-based CF: {}'.format(np.mean(rmses_mf)))

if __name__ == '__main__':
    main()
