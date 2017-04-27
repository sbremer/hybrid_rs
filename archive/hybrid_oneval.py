import pickle
import numpy as np

# np.random.seed(6)

# Local imports
import hybrid_model
from hybrid_model import HybridModel
import util

hybrid_model.verbose = 2
hybrid_model.batch_size = 500
hybrid_model.val_split = 0.1

# (meta_users, meta_items, U, I, Y) = pickle.load(open('data/ratings_metadata.pickle', 'wb'))
(meta_users, meta_items) = pickle.load(open('data/imdb_metadata.pickle', 'rb'))
(_, inds_u, inds_i, y) = pickle.load(open('data/cont.pickle', 'rb'))

inds_u = inds_u.astype(np.int)
inds_i = inds_i.astype(np.int)

# meta_items = meta_items[:, 0:19]

# Normalize features and set Nans to zero (=mean)
meta_users = (meta_users - np.nanmean(meta_users, axis=0)) / np.nanstd(meta_users, axis=0)
meta_items = (meta_items - np.nanmean(meta_items, axis=0)) / np.nanstd(meta_items, axis=0)
meta_users[np.isnan(meta_users)] = 0
meta_items[np.isnan(meta_items)] = 0

# Rescale ratings to ~(0.0, 1.0)
y_org = y.copy()
y = (y - 0.5) * 0.2

# Crossvalidation
n_fold = 5
user_coldstart = False
if user_coldstart:
    # kfold = util.kfold_entries(n_fold, inds_u)
    kfold = util.kfold_entries_plus(n_fold, inds_u, 3)
else:
    kfold = util.kfold(n_fold, inds_u)

xval_train, xval_test = next(kfold)

# Create model
model = HybridModel(meta_users, meta_items)

# Dataset training
inds_u_train = inds_u[xval_train]
inds_i_train = inds_i[xval_train]
y_train = y[xval_train]
n_train = len(y_train)

# Dataset testing
inds_u_test = inds_u[xval_test]
inds_i_test = inds_i[xval_test]
y_test = y[xval_test]

# mean = np.mean(y_train)
# y_train -= mean
# y_test -= mean

# Run initial (separate) training
model.train_initial(inds_u_train, inds_i_train, y_train, True)
print('Testing using test set before cross-training:')
rmse_mf, rmse_ann = model.test(inds_u_test, inds_i_test, y_test, True)

rmses_mf = [rmse_mf]
rmses_ann = [rmse_ann]

vloss_mf = []
vloss_ann = []

hybrid_model.batch_size = 1024
hybrid_model.val_split = 0.25

f_tsize = 1.0
f_xsize = 0.3

if user_coldstart:
    history_mf = model._step_mf(2.0)

    print('Results after coldstart xtrain step:')
    rmse_mf, rmse_ann = model.test(inds_u_test, inds_i_test, y_test, True)
    vloss_mf.append(min(history_mf.history['val_loss']))
    vloss_ann.append(0.0)

    rmses_mf.append(rmse_mf)
    rmses_ann.append(rmse_ann)

# ANN step
history_ann = model.step_ann(0.3, f_tsize, True)
vloss_ann.append(min(history_ann.history['val_loss']))

# MF step
history_mf = model._step_mf(0.5, f_tsize, True)
vloss_mf.append(min(history_mf.history['val_loss']))

# Test
print('Results after training step:')
rmse_mf, rmse_ann = model.test(inds_u_test, inds_i_test, y_test, True)

# Alternating cross-training
# for i in range(3):
#     print('Training step {}'.format(i + 1))
#
#     # ANN step
#     history_ann = model.step_ann(0.3, f_tsize, True)
#     vloss_ann.append(min(history_ann.history['val_loss']))
#
#     # MF step
#     history_mf = model.step_mf(0.5, f_tsize, True)
#     vloss_mf.append(min(history_mf.history['val_loss']))
#
#     # Test
#     print('Results after training step {}:'.format(i + 1))
#     rmse_mf, rmse_ann = model.test(inds_u_test, inds_i_test, y_test, True)
#
#     rmses_mf.append(rmse_mf)
#     rmses_ann.append(rmse_ann)

# # Alternating cross-training
# for i in range(5):
#     print('Training step {}'.format(i + 1))
#
#     # ANN step
#     history_ann = model.step_ann(f_xsize, f_tsize, True)
#     vloss_ann.append(min(history_ann.history['val_loss']))
#
#     # MF step
#     history_mf = model.step_mf(f_xsize, f_tsize, True)
#     vloss_mf.append(min(history_mf.history['val_loss']))
#
#     # Test
#     print('Results after training step {}:'.format(i + 1))
#     rmse_mf, rmse_ann = model.test(inds_u_test, inds_i_test, y_test, True)
#
#     rmses_mf.append(rmse_mf)
#     rmses_ann.append(rmse_ann)

# import matplotlib.pyplot as plt
#
# x = np.arange(len(rmses_mf))
# f, axarr = plt.subplots(2, sharex=True)
# axarr[0].plot(x, rmses_mf, 'r-', x, rmses_ann, 'b-')
# axarr[0].legend(['RMSE MF', 'RMSE ANN'])
# axarr[1].plot(x[1:], vloss_mf, 'r--', x[1:], vloss_ann, 'b--')
# axarr[1].legend(['VLOSS MF', 'VLOSS ANN'])
# f.show()

# Model min 5
# RMSE MF: 0.9160 	MAE: 0.7192
# RMSE ANN: 0.9404 	MAE: 0.7440
