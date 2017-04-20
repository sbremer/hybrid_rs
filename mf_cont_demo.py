import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

# Keras
from keras.layers import Input, Embedding, Dense, Flatten
from keras.layers.merge import Dot, Concatenate
from keras.models import Model
from keras.regularizers import l2

# Local
import util
from util import BiasLayer


def get_model_bias():

    lmdba = 0.00005
    regularizer = l2(lmdba)

    input_u = Input((1,))
    input_i = Input((1,))

    bias_u = Embedding(n_users, 1, input_length=1, embeddings_initializer='zeros', embeddings_regularizer=regularizer)(input_u)
    bias_u_r = Flatten()(bias_u)
    bias_i = Embedding(n_items, 1, input_length=1, embeddings_initializer='zeros', embeddings_regularizer=regularizer)(input_i)
    bias_i_r = Flatten()(bias_i)

    added = Concatenate()([bias_u_r, bias_i_r])
    bias_out = util.BiasLayer()(added)

    model = Model(inputs=[input_u, input_i], outputs=bias_out)

    model.compile(loss='mse', optimizer='nadam')

    return model


def get_model_ann(meta_users, meta_items):
    lmdba = 0.00005
    regularizer = l2(lmdba)

    input_u = Input((1,))
    input_i = Input((1,))

    n_users, features_user = meta_users.shape[:2]
    n_items, features_items = meta_items.shape[:2]

    vec_u = Embedding(n_users, features_user, input_length=1, trainable=False)(input_u)
    vec_u_r = Flatten()(vec_u)
    vec_i = Embedding(n_items, features_items, input_length=1, trainable=False)(input_i)
    vec_i_r = Flatten()(vec_i)

    vec_features = Concatenate()([vec_u_r, vec_i_r])

    ann_1 = Dense(100, kernel_initializer='uniform', activation='sigmoid')(vec_features)
    ann_2 = Dense(20, kernel_initializer='uniform', activation='sigmoid')(ann_1)
    ann_3 = Dense(1, kernel_initializer='uniform', activation='sigmoid')(ann_2)

    bias_u = Embedding(n_users, 1, input_length=1, embeddings_regularizer=regularizer)(input_u)
    bias_u_r = Flatten()(bias_u)
    bias_i = Embedding(n_items, 1, input_length=1, embeddings_regularizer=regularizer)(input_i)
    bias_i_r = Flatten()(bias_i)

    added = Concatenate()([bias_u_r, bias_i_r, ann_3])

    ann_out = BiasLayer()(added)

    model = Model(inputs=[input_u, input_i], outputs=ann_out)

    model.compile(loss='mse', optimizer='nadam')

    model.layers[2].set_weights([meta_users])
    model.layers[3].set_weights([meta_items])

    return model


def get_model_mf_cont_demo(n_factors, n_users_features, n_items_feature):
    lmdba = 0.00005
    regularizer = l2(lmdba)

    input_u = Input((1,))
    input_i = Input((1,))

    vec_u = Embedding(n_users, n_users_features, input_length=1, trainable=False, name='users_features')(input_u)
    vec_u = Flatten()(vec_u)

    factors_u = Dense(n_factors, kernel_initializer='normal', activation='linear',
                             kernel_regularizer=regularizer, use_bias=False)(vec_u)

    vec_i = Embedding(n_items, n_items_feature, input_length=1, trainable=False, name='items_features')(input_i)
    vec_i = Flatten()(vec_i)

    factors_i = Dense(n_factors, kernel_initializer='normal', activation='linear',
                      kernel_regularizer=regularizer, use_bias=False)(vec_i)

    mf = Dot(1)([factors_u, factors_i])

    bias_u = Embedding(n_users, 1, input_length=1, embeddings_initializer='zeros', embeddings_regularizer=regularizer)(input_u)
    bias_u = Flatten()(bias_u)
    bias_i = Embedding(n_items, 1, input_length=1, embeddings_initializer='zeros', embeddings_regularizer=regularizer)(input_i)
    bias_i = Flatten()(bias_i)

    concat = Concatenate()([bias_u, bias_i, mf])

    mf_out = BiasLayer()(concat)

    model = Model(inputs=[input_u, input_i], outputs=mf_out)

    # Compile and return model
    model.compile(loss='mse', optimizer='nadam')
    return model

# Load data
(inds_u, inds_i, y, users_features, items_features) = pickle.load(open('data/ml100k.pickle', 'rb'))

n_users, n_users_features = users_features.shape
n_items, n_items_features = items_features.shape

# Transform rating space
y = (y - 0.5) * 0.2

# Get models
model_bias = get_model_bias()
model_ann = get_model_ann(users_features, items_features)
model_mf_cd = get_model_mf_cont_demo(40, n_users_features, n_items_features)

# Apply normalization to features
users_features = users_features / np.sqrt(np.maximum(1, np.sum(users_features, axis=1)[:, None]))
items_features = items_features / np.sqrt(np.maximum(1, np.sum(items_features, axis=1)[:, None]))

# Set weights
model_mf_cd.get_layer('users_features').set_weights([users_features])
model_mf_cd.get_layer('items_features').set_weights([items_features])

# Crossvalidation
n_fold = 5
user_coldstart = True
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

# Dataset testing
inds_u_test = inds_u[xval_test]
inds_i_test = inds_i[xval_test]
y_test = y[xval_test]

# Training
callbacks = [util.EarlyStoppingBestVal('val_loss', patience=10, min_delta=0.0001)]

model_bias.fit([inds_u_train, inds_i_train], y_train, batch_size=500, epochs=100,
                   validation_split=0.1, verbose=2, callbacks=callbacks)

model_ann.fit([inds_u_train, inds_i_train], y_train, batch_size=500, epochs=100,
                   validation_split=0.1, verbose=2, callbacks=callbacks)

model_mf_cd.fit([inds_u_train, inds_i_train], y_train, batch_size=500, epochs=100,
                   validation_split=0.1, verbose=2, callbacks=callbacks)


def test(model, description):
    y = model.predict([inds_u_train, inds_i_train])
    rmse_train = sqrt(mean_squared_error(y_train * 5, y * 5))

    y = model.predict([inds_u_test, inds_i_test])
    rmse_test = sqrt(mean_squared_error(y_test * 5, y * 5))

    print('{}: Training {:.4f}  Testing {:.4f}'.format(description, rmse_train, rmse_test))

print('')
test(model_bias, 'Bias')
test(model_ann, 'ANN ')
test(model_mf_cd, 'MFCD')


