import numpy as np
from keras.layers import Embedding, Reshape, Input, Dense
from keras.layers.merge import Dot, Concatenate, Add
from keras.models import Model
from keras.callbacks import EarlyStopping

import keras
import pickle
from sklearn.metrics import mean_squared_error
from math import sqrt

import util

# Hardcoded for now
n_users = 943
n_items = 1682


def mf_model(n_factors=20, include_bias=False):
    lmdba = 0.00001
    regularizer = keras.regularizers.l2(lmdba)

    input_u = Input((1,))
    input_i = Input((1,))

    vec_u = Embedding(n_users, n_factors, input_length=1, embeddings_regularizer=regularizer)(input_u)
    vec_u_r = Reshape((n_factors,))(vec_u)
    vec_i = Embedding(n_items, n_factors, input_length=1, embeddings_regularizer=regularizer)(input_i)
    vec_i_r = Reshape((n_factors,))(vec_i)

    mf = Dot(1)([vec_u_r, vec_i_r])

    if include_bias:
        bias_u = Embedding(n_users, 1, input_length=1, embeddings_regularizer=regularizer)(input_u)
        bias_u_r = Reshape((1,))(bias_u)
        bias_i = Embedding(n_items, 1, input_length=1, embeddings_regularizer=regularizer)(input_i)
        bias_i_r = Reshape((1,))(bias_i)

        added = Add()([bias_u_r, bias_i_r, mf])

        model = Model(inputs=[input_u, input_i], outputs=added)
    else:
        model = Model(inputs=[input_u, input_i], outputs=mf)

    # Compile and return model
    model.compile(loss='mse', optimizer='adamax')
    return model


def ann_model(meta_users, meta_items):

    input_u = Input((1,))
    input_i = Input((1,))

    n_users, features_user = meta_users.shape[:2]
    n_items, features_items = meta_items.shape[:2]

    vec_u = Embedding(n_users, features_user, input_length=1, trainable=False)(input_u)
    vec_u_r = Reshape((features_user,))(vec_u)
    vec_i = Embedding(n_items, features_items, input_length=1, trainable=False)(input_i)
    vec_i_r = Reshape((features_items,))(vec_i)

    vec_features = Concatenate()([vec_u_r, vec_i_r])

    ann_1 = Dense(200, kernel_initializer='uniform', activation='sigmoid')(vec_features)
    ann_2 = Dense(50, kernel_initializer='uniform', activation='sigmoid')(ann_1)
    ann_3 = Dense(1, kernel_initializer='uniform', activation='sigmoid')(ann_2)

    model = Model(inputs=[input_u, input_i], outputs=ann_3)
    model.compile(loss='mse', optimizer='nadam')

    model.layers[2].set_weights([meta_users])
    model.layers[3].set_weights([meta_items])

    return model


def step_mf():
    Ux, Ix = indexgen.get_indices(int(n_train / 2))

    Yx = model_ann.predict([Ux, Ix]).flatten()

    Ut = np.concatenate((U, Ux))
    It = np.concatenate((I, Ix))
    Yt = np.concatenate((Y_scaled, Yx))

    history = model_mf.fit([Ut, It], Yt, batch_size=500, epochs=100, validation_split=0.1, verbose=2, callbacks=callbacks_mf)


def test_mf(prnt=False):
    y_pred = model_mf.predict([U, I]) / 0.2 + 0.5
    rmse = sqrt(mean_squared_error(Y, y_pred))
    if prnt:
        print('RMSE MF: {}'.format(rmse))
    return rmse


def step_ann():
    Ux, Ix = indexgen.get_indices(int(n_train / 2))

    Yx = model_mf.predict([Ux, Ix]).flatten()

    Ut = np.concatenate((U, Ux))
    It = np.concatenate((I, Ix))
    Yt = np.concatenate((Y_scaled, Yx))

    history = model_ann.fit([Ut, It], Yt, batch_size=500, epochs=100, validation_split=0.1, verbose=2, callbacks=callbacks_ann)


def test_ann(prnt=False):
    y_pred = model_ann.predict([U, I]) / 0.2 + 0.5
    rmse = sqrt(mean_squared_error(Y, y_pred))
    if prnt:
        print('RMSE ANN: {}'.format(rmse))
    return rmse


# (meta_users, meta_items, U, I, Y) = pickle.load(open('data/ratings_metadata.pickle', 'wb'))
(meta_users, meta_items) = pickle.load(open('data/imdb_metadata.pickle', 'rb'))
(_, U, I, Y) = pickle.load(open('data/cont.pickle', 'rb'))

# Normalize features and set Nans to zero (=mean)
meta_users = (meta_users - np.nanmean(meta_users, axis=0)) / np.nanstd(meta_users, axis=0)
meta_items = (meta_items - np.nanmean(meta_items, axis=0)) / np.nanstd(meta_items, axis=0)
meta_users[np.isnan(meta_users)] = 0
meta_items[np.isnan(meta_items)] = 0

# Rescale ratings to ~(0.0, 1.0)
Y_scaled = (Y - 0.5) * 0.2

# Early stopping callbacks. ANN validation varies more -> higher patience
callbacks_mf = [EarlyStopping('val_loss', patience=4)]
callbacks_ann = [EarlyStopping('val_loss', patience=10)]

# Create and train MF
model_mf = mf_model(include_bias=True)
history = model_mf.fit([U, I], Y_scaled, batch_size=500, epochs=100, validation_split=0.1, verbose=2, callbacks=callbacks_mf)
test_mf(True)

# Create and train ANN
model_ann = ann_model(meta_users, meta_items)
history = model_ann.fit([U, I], Y_scaled, batch_size=500, epochs=100, validation_split=0.1, verbose=2, callbacks=callbacks_ann)
test_ann(True)

indexgen = util.IndexGen(n_users, n_items, U, I)
n_train = len(U)

for iter in range(3):
    # MF step
    step_mf()
    test_mf(True)

    # ANN step
    step_ann()
    test_ann(True)

