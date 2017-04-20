import pickle
import numpy as np

np.random.seed(0)

import keras
from keras.layers import Embedding, Reshape, Input, Dense, Flatten
from keras.layers.merge import Concatenate, Add, Multiply, Dot
from keras.constraints import maxnorm
from keras.models import Model
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
    bias_u_r = Flatten()(bias_u)
    bias_i = Embedding(n_items, 1, input_length=1, embeddings_regularizer=regularizer)(input_i)
    bias_i_r = Flatten()(bias_i)

    added = Concatenate()([bias_u_r, bias_i_r])
    bias_out = util.BiasLayer()(added)

    model = Model(inputs=[input_u, input_i], outputs=bias_out)

    model.compile(loss='mse', optimizer='adam')

    return model


def get_model_mf(n_factors=20):
    lmdba = 0.00005
    regularizer = keras.regularizers.l2(lmdba)

    # Hardcoded for now
    n_users = 943
    n_items = 1682

    input_u = Input((1,))
    input_i = Input((1,))

    vec_u = Embedding(n_users, n_factors, input_length=1, embeddings_regularizer=regularizer)(input_u)
    vec_u_r = Reshape((n_factors,))(vec_u)
    vec_i = Embedding(n_items, n_factors, input_length=1, embeddings_regularizer=regularizer)(input_i)
    vec_i_r = Reshape((n_factors,))(vec_i)

    mf = Dot(1)([vec_u_r, vec_i_r])

    bias_u = Embedding(n_users, 1, input_length=1, embeddings_regularizer=regularizer)(input_u)
    bias_u_r = Reshape((1,))(bias_u)
    bias_i = Embedding(n_items, 1, input_length=1, embeddings_regularizer=regularizer)(input_i)
    bias_i_r = Reshape((1,))(bias_i)

    added = Concatenate()([bias_u_r, bias_i_r, mf])

    mf_out = util.BiasLayer()(added)

    model = Model(inputs=[input_u, input_i], outputs=mf_out)

    # Compile and return model
    model.compile(loss='mse', optimizer='nadam')
    return model


def get_model_mf_implicit(n_factors, implicit):
    lmdba = 0.00005
    lmdba = 0.00007
    regularizer = keras.regularizers.l2(lmdba)

    # Hardcoded for now
    n_users = 943
    n_items = 1682

    input_u = Input((1,))
    input_i = Input((1,))

    vec_u = Embedding(n_users, n_factors, input_length=1, embeddings_regularizer=regularizer)(input_u)
    vec_u_r = Flatten()(vec_u)
    vec_i = Embedding(n_items, n_factors, input_length=1, embeddings_regularizer=regularizer)(input_i)
    vec_i_r = Flatten()(vec_i)

    vec_implicit = Embedding(n_users, n_items, input_length=1, trainable=False)(input_u)
    implicit_factors = Dense(n_factors, kernel_initializer='normal', activation='linear', kernel_regularizer=regularizer)(vec_implicit)
    implicit_factors = Flatten()(implicit_factors)

    vec_u_added = Add()([vec_u_r, implicit_factors])

    mf = Dot(1)([vec_u_added, vec_i_r])

    bias_u = Embedding(n_users, 1, input_length=1, embeddings_regularizer=regularizer)(input_u)
    bias_u_r = Flatten()(bias_u)
    bias_i = Embedding(n_items, 1, input_length=1, embeddings_regularizer=regularizer)(input_i)
    bias_i_r = Flatten()(bias_i)

    added = Concatenate()([bias_u_r, bias_i_r, mf])

    mf_out = util.BiasLayer()(added)

    model = Model(inputs=[input_u, input_i], outputs=mf_out)

    model.layers[1].set_weights([implicit])

    # Compile and return model
    model.compile(loss='mse', optimizer='nadam')
    return model


def get_model_ann(meta_users, meta_items):
    lmdba = 0.00001
    regularizer = keras.regularizers.l2(lmdba)

    input_u = Input((1,))
    input_i = Input((1,))

    n_users, features_user = meta_users.shape[:2]
    n_items, features_items = meta_items.shape[:2]

    vec_u = Embedding(n_users, features_user, input_length=1, trainable=False)(input_u)
    vec_u_r = Flatten()(vec_u)
    vec_i = Embedding(n_items, features_items, input_length=1, trainable=False)(input_i)
    vec_i_r = Flatten()(vec_i)

    vec_features = Concatenate(trainable=False)([vec_u_r, vec_i_r])

    ann_1 = Dense(100, kernel_initializer='uniform', activation='sigmoid', kernel_constraint=maxnorm(2))(vec_features)
    # ann_1 = keras.layers.Dropout(0.4)(ann_1)
    ann_2 = Dense(20, kernel_initializer='uniform', activation='sigmoid', kernel_constraint=maxnorm(2))(ann_1)

    # k = 2
    # comb = util.InputCombinations(k, trainable=False)(vec_features)
    # from keras.layers import LocallyConnected1D, Flatten
    # ann_2 = LocallyConnected1D(1, 1)(comb)
    #
    # ann_2 = Flatten()(ann_2)

    ann_3 = Dense(1, kernel_initializer='uniform', activation='sigmoid')(ann_2)

    bias_u = Embedding(n_users, 1, input_length=1, embeddings_regularizer=regularizer)(input_u)
    bias_u_r = Flatten()(bias_u)
    bias_i = Embedding(n_items, 1, input_length=1, embeddings_regularizer=regularizer)(input_i)
    bias_i_r = Flatten()(bias_i)

    added = Add()([bias_u_r, bias_i_r, ann_3])

    model = Model(inputs=[input_u, input_i], outputs=added)

    model.compile(loss='mse', optimizer='nadam')

    model.layers[2].set_weights([meta_users])
    model.layers[3].set_weights([meta_items])

    return model


def get_model_ann_test(meta_users, meta_items):
    lmdba = 0.00001
    regularizer = keras.regularizers.l2(lmdba)

    input_u = Input((1,))
    input_i = Input((1,))

    n_users, features_user = meta_users.shape[:2]
    n_items, features_items = meta_items.shape[:2]

    vec_u = Embedding(n_users, features_user, input_length=1, trainable=False)(input_u)
    vec_u_r = Flatten()(vec_u)
    vec_i = Embedding(n_items, features_items, input_length=1, trainable=False)(input_i)
    vec_i_r = Flatten()(vec_i)

    ann_u_1 = Dense(100, kernel_initializer='uniform', activation='sigmoid')(vec_u_r)
    ann_u_2 = Dense(20, kernel_initializer='uniform', activation='sigmoid')(ann_u_1)

    ann_i_1 = Dense(100, kernel_initializer='uniform', activation='sigmoid')(vec_i_r)
    ann_i_2 = Dense(20, kernel_initializer='uniform', activation='sigmoid')(ann_i_1)

    # ann_mlp = Dot(1)([ann_u_2, ann_i_2])
    ann_combined = Concatenate()([ann_u_2, ann_i_2])

    ann_mlp = Dense(10, kernel_initializer='uniform', activation='sigmoid')(ann_combined)

    bias_u = Embedding(n_users, 1, input_length=1, embeddings_regularizer=regularizer)(input_u)
    bias_u_r = Flatten()(bias_u)
    bias_i = Embedding(n_items, 1, input_length=1, embeddings_regularizer=regularizer)(input_i)
    bias_i_r = Flatten()(bias_i)

    added = Concatenate()([bias_u_r, bias_i_r, ann_mlp])

    ann_out = Dense(1, kernel_initializer='uniform', activation='sigmoid')(added)

    model = Model(inputs=[input_u, input_i], outputs=ann_out)

    model.compile(loss='mse', optimizer='nadam')

    model.layers[2].set_weights([meta_users])
    model.layers[3].set_weights([meta_items])

    return model


def get_model_ann_combinatory(meta_users, meta_items):
    lmdba = 0.00001
    regularizer = keras.regularizers.l2(lmdba)

    input_u = Input((1,))
    input_i = Input((1,))

    n_users, features_user = meta_users.shape[:2]
    n_items, features_items = meta_items.shape[:2]

    vec_u = Embedding(n_users, features_user, input_length=1, trainable=False)(input_u)
    vec_u_r = Flatten()(vec_u)
    vec_i = Embedding(n_items, features_items, input_length=1, trainable=False)(input_i)
    vec_i_r = Flatten()(vec_i)

    vec_features = Concatenate()([vec_u_r, vec_i_r])

    # ann_1 = Dense(100, kernel_initializer='uniform', activation='sigmoid', kernel_constraint=maxnorm(2))(vec_features)
    # ann_1 = keras.layers.Dropout(0.4)(ann_1)
    # ann_2 = Dense(20, kernel_initializer='uniform', activation='sigmoid', kernel_constraint=maxnorm(2))(ann_1)

    k = 2
    comb = util.InputCombinations(k)(vec_features)

    from keras.layers import LocallyConnected1D
    ann_2 = LocallyConnected1D(10, 1, activation='sigmoid')(comb)

    ann_2 = Flatten()(ann_2)

    ann_3 = Dense(1, kernel_initializer='uniform', activation='sigmoid')(ann_2)

    bias_u = Embedding(n_users, 1, input_length=1, embeddings_regularizer=regularizer)(input_u)
    bias_u_r = Flatten()(bias_u)
    bias_i = Embedding(n_items, 1, input_length=1, embeddings_regularizer=regularizer)(input_i)
    bias_i_r = Flatten()(bias_i)

    added = Add()([bias_u_r, bias_i_r, ann_3])

    model = Model(inputs=[input_u, input_i], outputs=added)

    model.compile(loss='mse', optimizer='nadam')

    model.layers[2].set_weights([meta_users])
    model.layers[3].set_weights([meta_items])

    return model


def get_model_rf_regressor():
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(30, n_jobs=-1)

    return model


(meta_users, meta_items) = pickle.load(open('data/imdb_metadata.pickle', 'rb'))
(X, inds_u, inds_i, y) = pickle.load(open('data/cont.pickle', 'rb'))

inds_u = inds_u.astype(np.int)
inds_i = inds_i.astype(np.int)


n_users = 943
n_items = 1682

# Normalize features and set Nans to zero (=mean)
meta_users = (meta_users - np.nanmean(meta_users, axis=0)) / np.nanstd(meta_users, axis=0)
meta_items = (meta_items - np.nanmean(meta_items, axis=0)) / np.nanstd(meta_items, axis=0)
meta_users[np.isnan(meta_users)] = 0
meta_items[np.isnan(meta_items)] = 0

# Rescale ratings to ~(0.0, 1.0)
# y_org = y.copy()
y = (y - 0.5) * 0.2

# Crossvalidation
n_fold = 5
user_coldstart = False
if user_coldstart:
    kfold = util.kfold_entries(n_fold, inds_u)
    # kfold = util.kfold_entries_plus(n_fold, inds_u, 3)
else:
    kfold = util.kfold(n_fold, inds_u)

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

implicit = np.zeros((n_users, n_items))
ratings_user = np.zeros(n_users)

for u, i, r in zip(inds_u_train, inds_i_train, y_train):
    if r >= 0.5:
        implicit[u, i] = 1.0
        ratings_user[u] += 1

implicit = implicit / np.sqrt(np.maximum(1, ratings_user[:, None]))

callbacks = [util.EarlyStoppingBestVal('val_loss', patience=3, min_delta=0.0001)]

# Keras Parameters:
keras_params = {'batch_size': 500, 'epochs': 100, 'validation_split': 0.2, 'verbose': 2, 'callbacks': callbacks}

# Get models
models = []
models.append(('Bias Only', get_model_bias(), keras_params))
# models.append(('MF + Bias', get_model_mf(50), keras_params))
# models.append(('MF + Impl', get_model_mf_implicit(20, implicit), keras_params))
# models.append(('ANN+ Bias', get_model_ann(meta_users, meta_items), keras_params))
# models.append(('ANN  Test', get_model_ann_test(meta_users, meta_items), keras_params))
# models.append(('ANN Combi', get_model_ann_combinatory(meta_users, meta_items), keras_params))
# models.append(('RF Regres', get_model_rf_regressor(), {}))

mean = np.mean(y_train)
# y_train = y_train - mean
# y_test = y_test - mean

results = []

for description, model, params in models:
    model.fit([inds_u_train, inds_i_train], y_train, **params)

    y = model.predict([inds_u_train, inds_i_train])
    rmse_train = sqrt(mean_squared_error(y_train * 5, y * 5))

    y = model.predict([inds_u_test, inds_i_test])
    rmse_test = sqrt(mean_squared_error(y_test * 5, y * 5))

    results.append((description, rmse_train, rmse_test))

for description, rmse_train, rmse_test in results:
    print('{}: Training {:.4f}  Testing {:.4f}'.format(description, rmse_train, rmse_test))
