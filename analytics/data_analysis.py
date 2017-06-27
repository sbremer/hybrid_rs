import scripts
from hybrid_model import dataset
import pandas as pd
import numpy as np

ds = dataset.get_dataset('ml100k')
(inds_u, inds_i, y, users_features, items_features) = ds.data
(users_desc, items_desc, users_features_desc, items_features_desc) = dataset.get_dataset_desc('ml100k')

n_users = ds.n_users
n_items = ds.n_items

n_users_features = len(users_features_desc)
n_items_features = len(items_features_desc)

# Sanity checks
assert (n_users, n_users_features) == users_features.shape
assert (n_items, n_items_features) == items_features.shape

matrix = np.zeros((n_users, n_items))

for u, i, r in zip(inds_u, inds_i, y):
    matrix[u, i] = r

entries = len(matrix.nonzero()[0])
sparsity = float(entries)
sparsity /= (matrix.shape[0] * matrix.shape[1])
print('Number of users {}'.format(n_users))
print('Number of Items {}'.format(n_items))
print('Total valid entries {}'.format(entries))
print('Sparsity {:4.4f}%'.format(sparsity * 100))
items_per_user_avg = np.mean(np.sum((matrix != 0).astype(np.int), 1))
users_per_item_avg = np.mean(np.sum((matrix != 0).astype(np.int), 0))
print('Average number of items per user {}'.format(items_per_user_avg))
print('Average number of users per item {}'.format(users_per_item_avg))

users_witout_items = np.sum(np.sum((matrix != 0), 1) == 0)
print('Users without any items {}'.format(users_witout_items))

items_without_users = np.sum(np.sum((matrix != 0), 0) == 0)
print('Items without any users {}'.format(items_without_users))

# # Sparsify
# # Delete users and items without valid matrix
# sales_sparse = matrix[~np.all(matrix == 0, 1), :]
# sales_sparse = sales_sparse[:, ~np.all(matrix == 0, 0)]
#
# entries_sparse = len(sales_sparse.nonzero()[0])
# sparsity_sparse = float(entries)
# sparsity_sparse /= (sales_sparse.shape[0] * sales_sparse.shape[1])
# print('Removing users and items without matrix:')
# print('Number of users {}'.format(sales_sparse.shape[0]))
# print('Number of Items {}'.format(sales_sparse.shape[1]))
# print('Sparsity after removal of users without matrix {:4.4f}%'.format(sparsity_sparse * 100))
# items_per_user_avg = np.mean(np.sum((sales_sparse != 0), 1))
# users_per_item_avg = np.mean(np.sum((sales_sparse != 0), 0))
# print('Average number of items per user {}'.format(items_per_user_avg))
# print('Average number of users per item {}'.format(users_per_item_avg))

ga = np.mean(y)
y_shift = y - ga
print('Global Average: {:4.4f}'.format(ga))

user_stats = pd.DataFrame([], index=users_features_desc)
user_stats['# users'] = np.sum(users_features, 0)
user_stats['# interactions'] = np.sum(users_features[inds_u, :], 0)
user_stats['avg rating'] = users_features[inds_u, :].T @ y_shift / np.sum(users_features[inds_u, :], 0)
print(user_stats)

item_stats = pd.DataFrame([], index=items_features_desc)
item_stats['# items'] = np.sum(items_features, 0)
item_stats['# interactions'] = np.sum(items_features[inds_i, :], 0)
item_stats['avg rating'] = items_features[inds_i, :].T @ y_shift / np.sum(items_features[inds_i, :], 0)
print(item_stats)


user_feature_stats = pd.DataFrame((users_features.T @ users_features)[7:, :9], index=users_features_desc[7:], columns=users_features_desc[:9])
print(user_feature_stats)

item_feature_stats = pd.DataFrame((items_features.T @ items_features), index=items_features_desc, columns=items_features_desc)
print(item_feature_stats)

concat_features_desc = users_features_desc + items_features_desc
concat_features_interactions = np.concatenate((users_features[inds_u, :], items_features[inds_i, :]), 1)
feature_interactions = pd.DataFrame(concat_features_interactions.T @ concat_features_interactions, index=concat_features_desc, columns=concat_features_desc)
print(feature_interactions)

feature_corr = pd.DataFrame((concat_features_interactions.T * y_shift) @ concat_features_interactions / feature_interactions.values, index=concat_features_desc, columns=concat_features_desc)
print(feature_corr)
