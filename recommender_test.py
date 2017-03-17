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
    import numpy.ma as ma
    train_data_matrix_ma = ma.masked_values(train_data_matrix, 0)

    ratings_total = train_data_matrix_ma.mean()
    ratings_user = np.array(train_data_matrix_ma.mean(axis=1) - ratings_total)
    ratings_item = np.array(train_data_matrix_ma.mean(axis=0) - ratings_total)

    predicted_g = np.tile(ratings_total, (n_users, n_items))
    predicted_gui = ratings_total + np.tile(ratings_user, (n_items, 1)).T + np.tile(ratings_item, (n_users, 1))

    import implicit
    from scipy.sparse import coo_matrix

    train_data_matrix = train_data_matrix.T
    indices = np.nonzero(train_data_matrix)
    train_data_matrix = (train_data_matrix)
    cm = coo_matrix((train_data_matrix[indices], indices), shape=train_data_matrix.shape)

    import logging
    logging.basicConfig(level=logging.DEBUG)

    rank = 100
    model = implicit.als.AlternatingLeastSquares(factors=rank, num_threads=8, regularization=0.01, calculate_training_loss=True, iterations=10, use_native=False)
    # model.user_factors = np.random.rand(n_users, rank).astype(np.float64) * 0.01
    # model.item_factors = np.random.rand(n_items, rank).astype(np.float64) * 0.01
    model.fit(cm)

    # user_factors, item_factors = implicit.alternating_least_squares(cm, factors=500)
    predicted = model.user_factors.dot(model.item_factors.T)

    rmse_mf = rmse(predicted, test_data_matrix)

    print('MF CF MSE: ' + str(rmse_mf))
    return rmse_mf


def cf_mf2(train_data_matrix, test_data_matrix):
    lambda_ = 0.1
    n_factors = 10
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


def cf_mf3(train_data_matrix, test_data_matrix):
    R = train_data_matrix
    T = test_data_matrix
    # Index matrix for training data
    I = R.copy()
    I[I > 0] = 1
    I[I == 0] = 0

    # Index matrix for test data
    I2 = T.copy()
    I2[I2 > 0] = 1
    I2[I2 == 0] = 0

    # Calculate the RMSE
    def rmse(I, R, Q, P):
        return np.sqrt(np.sum((I * (R - np.dot(P.T, Q))) ** 2) / len(R[R > 0]))

    lmbda = 0.1  # Regularisation weight
    k = 20  # Dimensionality of latent feature space
    m, n = R.shape  # Number of users and items
    n_epochs = 15  # Number of epochs

    P = 3 * np.random.rand(k, m)  # Latent user feature matrix
    Q = 3 * np.random.rand(k, n)  # Latent movie feature matrix
    Q[0, :] = R[R != 0].mean(axis=0)  # Avg. rating for each movie
    E = np.eye(k)  # (k x k)-dimensional idendity matrix

    train_errors = []
    test_errors = []

    # Repeat until convergence
    for epoch in range(n_epochs):
        # Fix Q and estimate P
        for i, Ii in enumerate(I):
            nui = np.count_nonzero(Ii)  # Number of items user i has rated
            if (nui == 0): nui = 1  # Be aware of zero counts!

            # Least squares solution
            Ai = np.dot(Q, np.dot(np.diag(Ii), Q.T)) + lmbda * nui * E
            Vi = np.dot(Q, np.dot(np.diag(Ii), R[i].T))
            P[:, i] = np.linalg.solve(Ai, Vi)

        # Fix P and estimate Q
        for j, Ij in enumerate(I.T):
            nmj = np.count_nonzero(Ij)  # Number of users that rated item j
            if (nmj == 0): nmj = 1  # Be aware of zero counts!

            # Least squares solution
            Aj = np.dot(P, np.dot(np.diag(Ij), P.T)) + lmbda * nmj * E
            Vj = np.dot(P, np.dot(np.diag(Ij), R[:, j]))
            Q[:, j] = np.linalg.solve(Aj, Vj)

        train_rmse = rmse(I, R, Q, P)
        test_rmse = rmse(I2, T, Q, P)
        train_errors.append(train_rmse)
        test_errors.append(test_rmse)

        print("[Epoch %d/%d] train error: %f, test error: %f" \
        % (epoch + 1, n_epochs, train_rmse, test_rmse))

    print("Algorithm converged")
    return test_rmse


def cf_mf4(train_data_matrix, test_data_matrix):
    def alternating_least_squares(Cui, factors, regularization, iterations=20):
        users, items = Cui.shape

        X = np.random.rand(users, factors) * 0.01
        Y = np.random.rand(items, factors) * 0.01

        Ciu = Cui.T.tocsr()
        Cui = Cui.tocsr()
        for iteration in range(iterations):
            least_squares(Cui, X, Y, regularization)
            least_squares(Ciu, Y, X, regularization)

        return X, Y

    def nonzeros(m, row):
        """ returns the non zeroes of a row in csr_matrix """
        for index in range(m.indptr[row], m.indptr[row + 1]):
            yield m.indices[index], m.data[index]

    def least_squares(Cui, X, Y, regularization):
        users, factors = X.shape
        YtY = Y.T.dot(Y)

        for u in range(users):
            # accumulate YtCuY + regularization * I in A
            A = YtY + regularization * np.eye(factors)

            # accumulate YtCuPu in b
            b = np.zeros(factors)

            for i, confidence in nonzeros(Cui, u):
                factor = Y[i]
                A += (confidence - 1) * np.outer(factor, factor)
                b += confidence * factor

            # Xu = (YtCuY + regularization * I)^-1 (YtCuPu)
            X[u] = np.linalg.solve(A, b)

    import implicit
    from scipy.sparse import coo_matrix

    indices = np.nonzero(train_data_matrix)
    cm = coo_matrix((train_data_matrix[indices], indices), shape=train_data_matrix.shape)

    X, Y = alternating_least_squares(cm, 100, 0.1, 5)

    predicted = X.dot(Y.T)

    rmse_mf = rmse(predicted, test_data_matrix)
    print(rmse_mf)

    return rmse_mf


def cf_mf5(train_data_matrix, test_data_matrix):
    from ExplicitMF import ExplicitMF

    model = ExplicitMF(train_data_matrix, verbose=True)
    model.train()
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

        # from pyspark.mllib.recommendation import ALS
        # from pyspark.sql import SQLContext
        # from pyspark import SparkConf, SparkContext
        #
        # import os
        # os.environ["SPARK_HOME"] = "/opt/apache-spark/"
        #
        # conf = SparkConf() \
        #     .setAppName("MovieLensALS") \
        #     .set("spark.executor.memory", "2g")
        # sc = SparkContext(conf=conf)
        #
        # sqlCtx = SQLContext(sc)
        # spark_df = sqlCtx.createDataFrame(train.iloc[:,[0,1,2]])
        # model = ALS.train(spark_df.rdd, 100, 20, 0.01)

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
            rmse_mf = cf_mf5(train_data_matrix, test_data_matrix)

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
