import scripts
import numpy as np
import pandas as pd
import pickle

# Rating Data
ratings_header = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = pd.read_csv('data/ml-100k/u.data', sep='\t', names=ratings_header, encoding='latin-1')

# User Data
users_header = ['user_id', 'age', 'sex', 'occupation', 'zipcode']
users = pd.read_csv('data/ml-100k/u.user', sep='|', names=users_header, encoding='latin-1')

# Item Data
items_header = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url', 'unknown|0', 'Action|1',
                'Adventure|2', 'Animation|3', 'Children\'s|4', 'Comedy|5', 'Crime|6', 'Documentary|7', 'Drama|8',
                'Fantasy|9', 'Film-Noir|10', 'Horror|11', 'Musical|12', 'Mystery|13', 'Romance|14', 'Sci-Fi|15',
                'Thriller|16', 'War|17', 'Western|18']
items = pd.read_csv('data/ml-100k/u.item', sep='|', names=items_header, encoding='latin-1')

n_ratings = len(ratings)
n_users = ratings.user_id.unique().shape[0]
n_items = ratings.item_id.unique().shape[0]

# Build user data
age_groups = [0, 18, 25, 35, 45, 50, 56, 200]
user_age = np.array(users.age)

for lower, upper in zip(age_groups[:-1], age_groups[1:]):
    in_group = (lower <= user_age) & (user_age < upper)
    user_age[in_group] = lower

users_age_onehot = pd.get_dummies(user_age)
users_sex_onehot = pd.get_dummies(users.sex)
users_occ_onehot = pd.get_dummies(users.occupation)

users_features = pd.concat([users_age_onehot, users_sex_onehot, users_occ_onehot], axis=1).values

# Build item data
items_genre = items.iloc[:, 5:]

# items_age = items.iloc[:, 2]
# items_age = items_age.fillna('1995').apply(lambda x: x.split('-')[-1])

items_features = items_genre.values

inds_u = ratings.user_id.astype(np.int).values - 1
inds_i = ratings.item_id.astype(np.int).values - 1
y = ratings.rating.astype(np.float).values

pickle.dump((inds_u, inds_i, y, users_features, items_features), open('data/ml100k.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
