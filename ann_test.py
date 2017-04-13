import pickle
import numpy as np

np.random.seed(1)

import keras
from keras.layers import Embedding, Reshape, Input, Dense
from keras.layers.merge import Concatenate, Add
from keras.constraints import maxnorm
from keras.models import Model
from util import EarlyStoppingBestVal
from sklearn.metrics import mean_squared_error
from math import sqrt

# Local imports
import util


def get_model_bias():
    # Hardcoded for now
    n_users = 943
    n_items = 1682

    lmdba = 0.00001
    regularizer = keras.regularizers.l2(lmdba)

    input_u = Input((1,))
    input_i = Input((1,))

    bias_u = Embedding(n_users, 1, input_length=1, embeddings_regularizer=regularizer)(input_u)
    bias_u_r = Reshape((1,))(bias_u)
    bias_i = Embedding(n_items, 1, input_length=1, embeddings_regularizer=regularizer)(input_i)
    bias_i_r = Reshape((1,))(bias_i)

    added = Add()([bias_u_r, bias_i_r])

    model = Model(inputs=[input_u, input_i], outputs=added)

    model.compile(loss='mse', optimizer='adam')

    return model


def get_model_ann(meta_users, meta_items):
    lmdba = 0.00001
    regularizer = keras.regularizers.l2(lmdba)

    input_u = Input((1,))
    input_i = Input((1,))

    n_users, features_user = meta_users.shape[:2]
    n_items, features_items = meta_items.shape[:2]

    vec_u = Embedding(n_users, features_user, input_length=1, trainable=False)(input_u)
    vec_u_r = Reshape((features_user,))(vec_u)
    vec_i = Embedding(n_items, features_items, input_length=1, trainable=False)(input_i)
    vec_i_r = Reshape((features_items,))(vec_i)

    vec_features = Concatenate(trainable=False)([vec_u_r, vec_i_r])

    ann_1 = Dense(50, kernel_initializer='uniform', activation='sigmoid', kernel_constraint=maxnorm(2))(vec_features)
    # ann_1 = keras.layers.Dropout(0.4)(ann_1)
    ann_2 = Dense(10, kernel_initializer='uniform', activation='sigmoid', kernel_constraint=maxnorm(2))(ann_1)

    # k = 2
    # comb = util.InputCombinations(k, trainable=False)(vec_features)
    # from keras.layers import LocallyConnected1D, Flatten
    # ann_2 = LocallyConnected1D(1, 1)(comb)
    #
    # ann_2 = Flatten()(ann_2)

    ann_3 = Dense(1, kernel_initializer='uniform', activation='sigmoid')(ann_2)

    bias_u = Embedding(n_users, 1, input_length=1, embeddings_regularizer=regularizer)(input_u)
    bias_u_r = Reshape((1,))(bias_u)
    bias_i = Embedding(n_items, 1, input_length=1, embeddings_regularizer=regularizer)(input_i)
    bias_i_r = Reshape((1,))(bias_i)

    added = Add()([bias_u_r, bias_i_r, ann_3])

    model = Model(inputs=[input_u, input_i], outputs=added)

    model.compile(loss='mse', optimizer='adam')

    model.layers[2].set_weights([meta_users])
    model.layers[3].set_weights([meta_items])

    return model


(meta_users, meta_items) = pickle.load(open('data/imdb_metadata.pickle', 'rb'))
(_, inds_u, inds_i, y) = pickle.load(open('data/cont.pickle', 'rb'))

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
user_coldstart = True
if user_coldstart:
    kfold = util.kfold_entries_plus(n_fold, inds_u, 3)
else:
    kfold = util.kfold(n_fold, inds_u)

kfold = util.kfold_entries_plus(n_fold, inds_u, 1)

xval_train, xval_test = next(kfold)

# Dataset training
inds_u_train = inds_u[xval_train]
inds_i_train = inds_i[xval_train]
y_train = y[xval_train]
n_train = len(y_train)

# Dataset testing
inds_u_test = inds_u[xval_test]
inds_i_test = inds_i[xval_test]
y_test = y[xval_test]

# Get models
model_bias = get_model_bias()
model_ann = get_model_ann(meta_users, meta_items)

callbacks = [EarlyStoppingBestVal('val_loss', patience=3, min_delta=0.0001)]

mean = np.mean(y_train)

model_bias.fit([inds_u_train, inds_i_train], y_train - mean, batch_size=500, epochs=100,
                                     validation_split=0.2, verbose=2, callbacks=callbacks)

model_ann.fit([inds_u_train, inds_i_train], y_train, batch_size=500, epochs=100,
                                     validation_split=0.2, verbose=2, callbacks=callbacks)

# Calculate training error
y = model_bias.predict([inds_u_train, inds_i_train])
rmse_bias_train = sqrt(mean_squared_error((y_train - mean) * 5, y * 5))

y = model_ann.predict([inds_u_train, inds_i_train])
rmse_ann_train = sqrt(mean_squared_error(y_train * 5, y * 5))

# Calculate test set error
y = model_bias.predict([inds_u_test, inds_i_test])
rmse_bias_test = sqrt(mean_squared_error((y_test - mean) * 5, y * 5))

y = model_ann.predict([inds_u_test, inds_i_test])
rmse_ann_test = sqrt(mean_squared_error(y_test * 5, y * 5))

print('Bias Only: Training {:.4f}  Testing {:.4f}'.format(rmse_bias_train, rmse_bias_test))
print('ANN+ Bias: Training {:.4f}  Testing {:.4f}'.format(rmse_ann_train, rmse_ann_test))
