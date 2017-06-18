import scripts
import numpy as np
import results.plots as lplot
import matplotlib.pyplot as plt

from hybrid_model.dataset import get_dataset
from hybrid_model.index_sampler import IndexSamplerUserItembased as IndexSampler

dataset = get_dataset('ml100k')
(inds_u, inds_i, y, users_features, items_features) = dataset.data

# mat = np.zeros((dataset.n_users, dataset.n_items), np.float)
#
# for u, i in zip(inds_u, inds_i):
#     mat[u, i] = 1.0
#
# # Get user/item distributions and order
# dist_users = np.sum(mat, axis=1).astype(np.int)
# dist_items = np.sum(mat, axis=0).astype(np.int)

user_dist = np.bincount(inds_u, minlength=dataset.n_users)
item_dist = np.bincount(inds_i, minlength=dataset.n_items)

order_users = np.argsort(-user_dist)
order_items = np.argsort(-item_dist)

dist_users = user_dist[order_users]
dist_items = item_dist[order_items]

inds_u = np.argsort(order_users)[inds_u]
inds_i = np.argsort(order_items)[inds_i]

# Index sampling
sampler_config = {'f_cf': 0.15, 'min_ratings_user': 30, 'f_user': 3.0, 'min_ratings_item': 10, 'f_item': 3.0}
sampler = IndexSampler(dist_users, dist_items, sampler_config, [inds_u, inds_i])
from_cf = sampler.get_indices_from_cf()
from_md = sampler.get_indices_from_md()

from_cf = (from_cf[0].flatten(), from_cf[1].flatten())
from_md = (from_md[0].flatten(), from_md[1].flatten())

# Actual painting
# img = np.ones((dataset.n_users, dataset.n_items, 3), np.float)
# bs = 3
# for u, i in zip(inds_u, inds_i):
#     img[u-bs:u+bs, i-bs:i+bs, :] = 0.0
# bs = 2
# Paint samples
# for u, i in zip(*from_cf):
#     img[u-bs:u+bs, i-bs:i+bs, :] = np.array([0, 1, 0])
#
# for u, i in zip(*from_md):
#     img[u-bs:u+bs, i-bs:i+bs, :] = np.array([1, 0, 0])
#
#
# img = imresize(img, 1.0)
# for c in range(3):
#     img[:, :, c] = gaussian_filter(img[:, :, c], 1)

# plt.imshow(img)

# mat = np.zeros((dataset.n_users, dataset.n_items), np.float)
#
# for u, i in zip(inds_u, inds_i):
#     mat[u, i] = 1.0
#
#
# bs = 1
# mat = np.ones((dataset.n_users, dataset.n_items), np.float) * 0.5
# for u, i in zip(from_cf[0], from_cf[1]):
#     mat[u-bs:u+bs, i-bs:i+bs] = 1.0
#
# for u, i in zip(from_md[0], from_md[1]):
#     mat[u-bs:u+bs, i-bs:i+bs] = 0.0
#
# from scipy.ndimage.filters import gaussian_filter
#
# mat = gaussian_filter(mat, 1)
#
# im = plt.imshow(mat.transpose(), interpolation='sinc', origin='lower', cmap='bwr')
# plt.scatter(from_cf[1], from_cf[0], s=1, marker='_', alpha=0.7)
# plt.scatter(from_md[1], from_md[0], s=1, marker='|', alpha=0.7)
# plt.show()

fig, ax = lplot.newfig(0.99, 0.7)

plt.style.use('acm-1col')
ax.scatter(from_cf[0], from_cf[1], s=0.02, marker='_', label='$S_{CF}$', alpha=0.5)
ax.scatter(from_md[0], from_md[1], s=0.02, marker='|', label='$S_{MD}$', alpha=0.5)

ax.set_title('Index tuple sampling')
ax.set_xlabel('$\leftarrow$ more - less $\\to$\n\#ratings / user')
ax.set_ylabel('\#ratings / item\n$\leftarrow$ more - less $\\to$')

plt.xticks([], [])
plt.yticks([], [])

# ax.legend()
lgnd = ax.legend(loc="lower center", numpoints=1, fontsize=7)

#change the marker size manually for both lines
lgnd.legendHandles[0].set_sizes([20.0])
lgnd.legendHandles[1].set_sizes([20.0])

lplot.savefig('sampling')
# plt.show()
