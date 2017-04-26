import pickle
import numpy as np

# np.random.seed(6)

# Local imports
import hybrid_model_attbias
from hybrid_model_attbias import HybridModel
import util

hybrid_model_attbias.verbose = 2

(inds_u, inds_i, y, users_features, items_features) = pickle.load(open('data/ml100k.pickle', 'rb'))

n_users, n_users_features = users_features.shape
n_items, n_items_features = items_features.shape

# items_features_norm = items_features / np.maximum(1, np.sum(items_features, axis=1)[:, None])

# Rescale ratings to ~(0.0, 1.0)
y = (y - 0.5) * 0.2

# Crossvalidation
n_fold = 5
user_coldstart = True
if user_coldstart:
    kfold = util.kfold_entries(n_fold, inds_u)
    # kfold = util.kfold_entries_plus(n_fold, inds_u, 3)
else:
    kfold = util.kfold(n_fold, inds_u)

xval_train, xval_test = next(kfold)

# Create model
model = HybridModel(users_features, items_features)

# Dataset training
inds_u_train = inds_u[xval_train]
inds_i_train = inds_i[xval_train]
y_train = y[xval_train]
n_train = len(y_train)

# Dataset testing
inds_u_test = inds_u[xval_test]
inds_i_test = inds_i[xval_test]
y_test = y[xval_test]

# Run initial (separate) training
model.train_initial(inds_u_train, inds_i_train, y_train, True)
print('Testing using test set before cross-training:')
rmse_mf, rmse_ann = model.test(inds_u_test, inds_i_test, y_test, True)

rmses_mf = [rmse_mf]
rmses_ann = [rmse_ann]

vloss_mf = []
vloss_ann = []

f_tsize = 1.0
f_xsize = 0.3

if user_coldstart:
    history_mf = model.step_mf(2.0)

    print('Results after coldstart xtrain step:')
    rmse_mf, rmse_ann = model.test(inds_u_test, inds_i_test, y_test, True)
    vloss_mf.append(min(history_mf.history['val_loss']))
    vloss_ann.append(0.0)

    rmses_mf.append(rmse_mf)
    rmses_ann.append(rmse_ann)

# # ANN step
# history_ann = model.step_ann(0.2, f_tsize, True)
# vloss_ann.append(min(history_ann.history['val_loss']))
#
# # MF step
# history_mf = model.step_mf(0.2, f_tsize, True)
# vloss_mf.append(min(history_mf.history['val_loss']))
#
# # Test
# print('Results after training step {}:'.format(i+1))
# rmse_mf, rmse_ann = model.test(inds_u_test, inds_i_test, y_test, True)


# model.model_mf.compile(loss='mse', optimizer='adadelta')
# model.model_ann.compile(loss='mse', optimizer='adadelta')
#
# model.model_mf.compile(loss='mse', optimizer='sgd')
# model.model_ann.compile(loss='mse', optimizer='sgd')

# history_mf = model.step_cs(0.2, f_tsize, True)
# rmse_mf, rmse_ann = model.test(inds_u_test, inds_i_test, y_test, True)
#
# history_mf = model.step_mf(2.2, f_tsize, True)
# rmse_mf, rmse_ann = model.test(inds_u_test, inds_i_test, y_test, True)


# Alternating cross-training
for i in range(5):
    print('Training step {}'.format(i + 1))

    # ANN step
    history_ann = model.step_cs(0.3, f_tsize, True)
    vloss_ann.append(min(history_ann.history['val_loss']))

    # MF step
    history_mf = model.step_mf(0.3, f_tsize, True)
    vloss_mf.append(min(history_mf.history['val_loss']))

    # Test
    print('Results after training step {}:'.format(i + 1))
    rmse_mf, rmse_ann = model.test(inds_u_test, inds_i_test, y_test, True)

    rmses_mf.append(rmse_mf)
    rmses_ann.append(rmse_ann)

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

import matplotlib.pyplot as plt

x = np.arange(len(rmses_mf))
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(x, rmses_mf, 'r-', x, rmses_ann, 'b-')
axarr[0].legend(['RMSE MF', 'RMSE ANN'])
axarr[1].plot(x[1:], vloss_mf, 'r--', x[1:], vloss_ann, 'b--')
axarr[1].legend(['VLOSS MF', 'VLOSS ANN'])
f.show()
