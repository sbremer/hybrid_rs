import pickle
import numpy as np

np.random.seed(1)

import keras
from keras.layers import Embedding, Input, Dense, Flatten
from keras.layers.merge import Concatenate, Add, Dot
from keras.constraints import maxnorm
from keras.models import Model
from sklearn.metrics import mean_squared_error
from math import sqrt

# Local imports
import util


def get_model_mf_implicit(n_factors, implicit):
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

    vec_implicit = Embedding(n_users, n_items, input_length=1, name='implicit', trainable=False)(input_u)
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

    model.get_layer('implicit').set_weights([implicit])

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

    vec_u = Embedding(n_users, features_user, input_length=1, name='features_user', trainable=False)(input_u)
    vec_u_r = Flatten()(vec_u)
    vec_i = Embedding(n_items, features_items, input_length=1, name='features_items', trainable=False)(input_i)
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

    model.get_layer('features_user').set_weights([meta_users])
    model.get_layer('features_items').set_weights([meta_items])

    return model


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

n_factors = 20
lmdba = 0.00007
regularizer = keras.regularizers.l2(lmdba)

n_users, features_user = meta_users.shape[:2]
n_items, features_items = meta_items.shape[:2]


implicit = np.zeros((n_users, n_items))

lookup = {}

# Use ratings over the threshold as implicit feedback
for u, i, r in zip(inds_u_train, inds_i_train, y_train):
    if r >= 0.3:
        implicit[u, i] = 1.0

    lookup[(u, i)] = True

# Normalize using sqrt (ref. SVD++ paper)
implicit_norm = implicit / np.sqrt(np.maximum(1, np.sum(implicit, axis=1)[:, None]))

user_ratings = np.sum(implicit, axis=1).astype(np.int)

# User Cutoff
user_ratings_cut = int(np.median(user_ratings))

# Generate xtrain indicies
user_xtrain = np.maximum(0, user_ratings_cut - user_ratings)

n_xtrain = int(np.sum(user_xtrain))
inds_u_xtrain = np.zeros(n_xtrain, np.int)
inds_i_xtrain = np.zeros(n_xtrain, np.int)

at = 0
for u, user_n_xtrain in enumerate(user_xtrain):
    got = 0
    lookup_samples = {}
    while got < user_n_xtrain:
        i = np.random.randint(n_items)

        if (u, i) not in lookup and (u, i) not in lookup_samples:
            inds_u_xtrain[at] = u
            inds_i_xtrain[at] = i
            lookup_samples[(u, i)] = True
            got += 1
            at += 1

# Split test data (short/long-tail)
user_longtail = user_xtrain > 0
samples_test_longtail = user_longtail[inds_u_test]

inds_u_test_longtail = inds_u_test[samples_test_longtail]
inds_i_test_longtail = inds_i_test[samples_test_longtail]
y_test_longtail = y_test[samples_test_longtail]

inds_u_test_shorttail = inds_u_test[~samples_test_longtail]
inds_i_test_shorttail = inds_i_test[~samples_test_longtail]
y_test_shorttail = y_test[~samples_test_longtail]

# Create models
model_mf = get_model_mf_implicit(20, implicit_norm)
model_ann = get_model_ann(meta_users, meta_items)

# Training
callbacks = [util.EarlyStoppingBestVal('val_loss', patience=5, min_delta=0.0001)]


def test_ann():

    y = model_ann.predict([inds_u_train, inds_i_train])
    rmse_train = sqrt(mean_squared_error(y_train * 5, y * 5))

    y = model_ann.predict([inds_u_test, inds_i_test])
    rmse_test = sqrt(mean_squared_error(y_test * 5, y * 5))

    print('{}: Training {:.4f}  Testing {:.4f}'.format('ANN', rmse_train, rmse_test))

    if not user_coldstart:

        y = model_ann.predict([inds_u_test_shorttail, inds_i_test_shorttail])
        rmse_test_shorttail = sqrt(mean_squared_error(y_test_shorttail * 5, y * 5))

        y = model_ann.predict([inds_u_test_longtail, inds_i_test_longtail])
        rmse_test_longtail = sqrt(mean_squared_error(y_test_longtail * 5, y * 5))

        print('{}: Test(short) {:.4f}  Test(long) {:.4f}'.format('ANN', rmse_test_shorttail, rmse_test_longtail))


def test_mf():
    y = model_mf.predict([inds_u_train, inds_i_train])
    rmse_train = sqrt(mean_squared_error(y_train * 5, y * 5))

    y = model_mf.predict([inds_u_test, inds_i_test])
    rmse_test = sqrt(mean_squared_error(y_test * 5, y * 5))

    print('{}: Training {:.4f}  Testing {:.4f}'.format('MF', rmse_train, rmse_test))

    if not user_coldstart:

        y = model_mf.predict([inds_u_test_shorttail, inds_i_test_shorttail])
        rmse_test_shorttail = sqrt(mean_squared_error(y_test_shorttail * 5, y * 5))

        y = model_mf.predict([inds_u_test_longtail, inds_i_test_longtail])
        rmse_test_longtail = sqrt(mean_squared_error(y_test_longtail * 5, y * 5))

        print('{}: Test(short) {:.4f}  Test(long) {:.4f}'.format('MF', rmse_test_shorttail, rmse_test_longtail))


model_mf.fit([inds_u_train, inds_i_train], y_train, batch_size=500, epochs=100,
                   validation_split=0.1, verbose=2, callbacks=callbacks)

model_ann.fit([inds_u_train, inds_i_train], y_train, batch_size=500, epochs=100,
                   validation_split=0.1, verbose=2, callbacks=callbacks)

print(' ')
test_mf()
test_ann()

# Get generated xtrain ratings from ANN
y_xtrain = model_ann.predict([inds_u_xtrain, inds_i_xtrain])

y_xtrain = y_xtrain.flatten()

inds_u_xtrain_con = np.concatenate((inds_u_train, inds_u_xtrain))
inds_i_xtrain_con = np.concatenate((inds_i_train, inds_i_xtrain))
y_xtrain_con = np.concatenate((y_train, y_xtrain))

# Shuffle xtrain data
order = np.arange(len(y_xtrain_con))
np.random.shuffle(order)
inds_u_xtrain_con = inds_u_xtrain_con[order]
inds_i_xtrain_con = inds_i_xtrain_con[order]
y_xtrain_con = y_xtrain_con[order]

# Update implicit
for u, i, r in zip(inds_u_xtrain, inds_i_xtrain, y_xtrain):
    if r >= 0.5:
        implicit[u, i] = 1.0

implicit_norm = implicit / np.sqrt(np.maximum(1, np.sum(implicit, axis=1)[:, None]))
model_mf.get_layer('implicit').set_weights([implicit_norm])

# model_cf.fit([inds_u_xtrain_con, inds_i_xtrain_con], y_xtrain_con, batch_size=500, epochs=100,
#                    validation_split=0.1, verbose=0, callbacks=callbacks)
model_mf.fit([inds_u_train, inds_i_train], y_train, batch_size=500, epochs=100,
                   validation_split=0.1, verbose=0, callbacks=callbacks)

print(' ')
test_mf()
