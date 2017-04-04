import numpy as np
import pandas as pd
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt


def main():
    ratings_header = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv('data/ml-100k/u.data', sep='\t', names=ratings_header)

    n_ratings = len(ratings)
    n_users = ratings.user_id.unique().shape[0]
    n_items = ratings.item_id.unique().shape[0]
    print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

    (contentX, contentY) = pickle.load(open('data/cont.pickle', 'rb'))

    n_fold = 5
    kf = KFold(n_splits=n_fold, shuffle=True)

    rmses = []

    for train_indices, test_indices in kf.split(contentY):
        X_train = contentX[train_indices, :]
        Y_train = contentY[train_indices] / 6

        X_test = contentX[test_indices, :]
        Y_test = contentY[test_indices]

        model = MLPRegressor(hidden_layer_sizes=(100,50) )
        # model = LogisticRegression()
        model.fit(X_train, Y_train)

        Y_pred = model.predict(X_test) * 6

        rmse = sqrt(mean_squared_error(Y_pred, Y_test))
        rmses.append(rmse)

    print('Crossval RMSE of Content-based RS: {}'.format(np.mean(rmses)))

if __name__ == '__main__':
    main()