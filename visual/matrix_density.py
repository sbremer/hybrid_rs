import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter
import os

from hybrid_model.dataset import get_dataset
from hybrid_model.index_sampler import IndexSamplerUserbased

os.chdir('../')

dataset = get_dataset('ml100k')
(inds_u, inds_i, y, users_features, items_features) = dataset.data

mat = np.zeros((dataset.n_users, dataset.n_items), np.float)

for u, i in zip(inds_u, inds_i):
    mat[u, i] = 1.0

# Get user/item distributions and order
dist_users = np.sum(mat, axis=1).astype(np.int)
dist_items = np.sum(mat, axis=0).astype(np.int)

order_users = np.argsort(-dist_users)
order_items = np.argsort(-dist_items)

dist_users = dist_users[order_users]
dist_items = dist_items[order_items]

inds_u = np.argsort(order_users)[inds_u]
inds_i = np.argsort(order_items)[inds_i]

# Index sampling
sampler = IndexSamplerUserbased(dist_users, dist_items, [inds_u, inds_i])
from_mf = sampler.get_indices_from_mf()
from_cs = sampler.get_indices_from_cs()

# Actual painting
img = np.zeros((dataset.n_users, dataset.n_items, 3), np.float)

bs = 3

for u, i in zip(inds_u, inds_i):
    img[u-bs:u+bs, i-bs:i+bs, :] = 1.0

bs = 2

# Paint samples
for u, i in zip(*from_mf):
    img[u-bs:u+bs, i-bs:i+bs, :] = np.array([0, 1, 0])

for u, i in zip(*from_cs):
    img[u-bs:u+bs, i-bs:i+bs, :] = np.array([1, 0, 0])


img = imresize(img, 1.0)
for c in range(3):
    img[:, :, c] = gaussian_filter(img[:, :, c], 1)

plt.imshow(img)
plt.show()
