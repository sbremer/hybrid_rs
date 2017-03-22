import numpy as np
import pandas as pd
import pickle
import math


def main():
    ratings_header = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv('data/ml-100k/u.data', sep='\t', names=ratings_header)

    n_ratings = len(ratings)
    n_users = ratings.user_id.unique().shape[0]
    n_items = ratings.item_id.unique().shape[0]
    print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

    # Users from preprocessed user file with locations
    # ['user_id', 'age', 'sex', 'occupation', 'zip_code', 'loc_lon', 'loc_lat']

    users = pickle.load(open('data/users.pickle', 'rb'))

    n_features_user = 4  # Age, Sex, Location (lon, lat)
    n_occupations = users.occupation.unique().shape[0]
    n_features_user +=  n_occupations   # HotOne encoded

    occupation = pd.get_dummies(users.iloc[:, 3])

    # Item Data
    items_header = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url', 'unknown|0', 'Action|1', 'Adventure|2', 'Animation|3', 'Children\'s|4', 'Comedy|5', 'Crime|6', 'Documentary|7', 'Drama|8', 'Fantasy|9', 'Film-Noir|10', 'Horror|11', 'Musical|12', 'Mystery|13', 'Romance|14', 'Sci-Fi|15', 'Thriller|16', 'War|17', 'Western|18']
    items = pd.read_csv('data/ml-100k/u.item', sep='|', names=items_header, encoding='latin-1')

    n_features_item = 19  # Genre Tags

    X = np.zeros((n_ratings, (n_features_user + n_features_item)))
    U = np.zeros(n_ratings)
    I = np.zeros(n_ratings)
    Y = np.zeros(n_ratings)


    n_invalid = 0

    id = 0

    for rating in ratings.itertuples():
        # id = rating.Index
        userid = rating[1]
        itemid = rating[2]
        rating_score = rating[3]

        if math.isnan(users.iloc[userid - 1, 5]):
            n_invalid += 1
            continue

        U[id] = userid - 1
        I[id] = itemid - 1

        X[id, 0] = users.iloc[userid - 1, 1]  # Age
        X[id, 1] = -1 if users.iloc[userid - 1, 2] == 'M' else 1  # Sex
        X[id, 2:4] = users.iloc[userid - 1, [5, 6]]  # Location (lon, lat)
        X[id, 4:4+n_occupations] = occupation.iloc[userid - 1, :]
        X[id, 25:] = items.iloc[itemid - 1, 5:]

        Y[id] = rating_score

        id += 1

    X = X[:-n_invalid, :]
    Y = Y[:-n_invalid]
    U = U[:-n_invalid]
    I = I[:-n_invalid]

    print('{} invalid samples'.format(n_invalid))

    pickle.dump((X, U, I, Y), open('data/cont.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
