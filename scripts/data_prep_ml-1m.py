import scripts
import numpy as np
import pandas as pd
import pickle

# Rating Data
ratings_header = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = pd.read_csv('data/ml-1m/ratings.dat', sep='::', names=ratings_header, encoding='latin-1')

# User Data
users_header = ['user_id', 'sex', 'age', 'occupation', 'zipcode']
users = pd.read_csv('data/ml-1m/users.dat', sep='::', names=users_header, encoding='latin-1')

# Item Data
items_header = ['item_id', 'title', 'genre']
items = pd.read_csv('data/ml-1m/movies.dat', sep='::', names=items_header, encoding='latin-1')

n_ratings = len(ratings)
n_users = ratings.user_id.unique().shape[0]
n_items = max(ratings.item_id.unique())

users_age_onehot = pd.get_dummies(users.age)
users_sex_onehot = pd.get_dummies(users.sex)
users_occ_onehot = pd.get_dummies(users.occupation)

occupations = ['other', 'academic/educator', 'artist', 'clerical/admin', 'college/grad student', 'customer service',
               'doctor/health care', 'executive/managerial', 'farmer', 'homemaker', 'K-12 student', 'lawyer',
               'programmer', 'retired', 'matrix/marketing', 'scientist', 'self-employed', 'technician/engineer',
               'tradesman/craftsman', 'unemployed', 'writer']

desc_age = list(map(str, users_age_onehot.columns.tolist()))
desc_sex = users_sex_onehot.columns.tolist()
desc_occ = occupations

users_features = pd.concat([users_age_onehot, users_sex_onehot, users_occ_onehot], axis=1).values.astype(np.int)

users_features_desc = desc_age + desc_sex + desc_occ

movies_empty = list(set(np.arange(1, n_items + 1)) - set(items.item_id.unique()))
d = {'item_id': movies_empty, 'title': [''] * len(movies_empty), 'genre': [''] * len(movies_empty)}
items = items.append(pd.DataFrame(d))
items = items.iloc[items.item_id.argsort().values, :]

items_genre = items.genre.str.get_dummies(sep='|')
items_features = items_genre.values.astype(np.int)

items_features_desc = items_genre.columns.tolist()

users_desc = list(map(lambda x: 'User_{}'.format(x), range(n_users)))
items_desc = items['title'].tolist()

inds_u = ratings.user_id.astype(np.int).values - 1
inds_i = ratings.item_id.astype(np.int).values - 1
y = ratings.rating.astype(np.float).values

pickle.dump((inds_u, inds_i, y, users_features, items_features), open('data/ml1m.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
pickle.dump((users_desc, items_desc, users_features_desc, items_features_desc), open('data/ml1m_desc.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
