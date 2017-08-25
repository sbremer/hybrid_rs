import script_chdir
import numpy as np
import results.plots as lplot
import matplotlib.pyplot as plt

users = 40
items = 60
data = np.ones((users, items, 3))

samples = 130

set = np.random.choice(users*items, samples, replace=False)

c_train = [1.0, 0.0, 0.0]
c_test = [0.0, 1.0, 0.0]

for i, s in enumerate(set):
    user = int(s / items)
    item = s % items

    if item < items * 0.8:
        data[user, item, :] = c_train
    else:
        data[user, item, :] = c_test

fig, ax = lplot.newfig(1.0)

ax.imshow(data)

ax.set_title('Cross-Validation Item cold-start')
ax.set_xlabel('Items')
ax.set_ylabel('Users')

import matplotlib.patches as mpatches

red_patch = mpatches.Patch(color=c_train, label='Training')
green_patch = mpatches.Patch(color=c_test, label='Testing')
lgnd = ax.legend(handles=[red_patch, green_patch], loc="lower right", numpoints=1, fontsize=12)

plt.xticks([], [])
plt.yticks([], [])

lplot.savefig('xval_ics')
