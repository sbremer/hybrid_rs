import scripts
import numpy as np
import results.plots as lplot
import matplotlib.pyplot as plt

from hybrid_model.dataset import get_dataset
from hybrid_model.index_sampler import IndexSamplerUserItembased as IndexSampler

dataset = get_dataset('ml100k')
(inds_u, inds_i, y, users_features, items_features) = dataset.data

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

fig, ax = lplot.newfig(1.0, 0.7)

# Got color through next(ax._get_lines.prop_cycler)
ax.scatter(from_md[0], from_md[1], s=0.02, color='#ff7f0e', marker='s', label='$S_{MD}$', alpha=0.5)

ax.set_title('Index Tuple Sampling from MD')
ax.set_xlabel('Users with\n$\leftarrow$ more - fewer $\\to$\nratings')
ax.set_ylabel('Items with\n$\leftarrow$ more - fewer $\\to$\nratings')

plt.xticks([], [])
plt.yticks([], [])

lgnd = ax.legend(loc="lower center", numpoints=1, fontsize=7)

#change the marker size manually for both lines
for handle in lgnd.legendHandles:
    handle._alpha = 1.0
    handle.set_sizes([20])

lplot.savefig('sampling_from_md')
# plt.show()
