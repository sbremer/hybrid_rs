import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

np.random.seed(0)

# Keras
from keras.layers import Input, Embedding, Dense, Flatten
from keras.layers.merge import Dot, Concatenate, Add
from keras.models import Model
from keras.regularizers import l2

# Local
import util
from util import BiasLayer


def get_model_bias():

    lmdba = 0.00007
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


def get_model_bias_item():

    lmdba = 0.00007
    regularizer = l2(lmdba)

    input_u = Input((1,))
    input_i = Input((1,))

    bias_i = Embedding(n_items, 1, input_length=1, embeddings_initializer='zeros', embeddings_regularizer=regularizer)(input_i)
    bias_i_r = Flatten()(bias_i)

    bias_out = util.BiasLayer()(bias_i_r)

    model = Model(inputs=[input_u, input_i], outputs=bias_out)

    model.compile(loss='mse', optimizer='nadam')

    return model


def get_model_bias_custom(include_user=True, include_item=True):

    class BiasEstimator:
        def __init__(self, include_user=True, include_item=True):
            self.include_user = include_user
            self.include_item = include_item

        def fit(self, x, y, **kwargs):
            self.global_avg = 0.0
            self.bias_user = np.zeros((n_users,))
            self.bias_item = np.zeros((n_items,))

            user_r = np.zeros((n_users,))
            item_r = np.zeros((n_items,))

            inds_u = x[0]
            inds_i = x[1]

            for u, i, r in zip(inds_u, inds_i, y):
                self.global_avg += r
                self.bias_user[u] += r
                user_r[u] += 1.0
                self.bias_item[i] += r
                item_r[i] += 1.0

            self.global_avg /= len(y)

            if self.include_user:
                self.bias_user = self.bias_user / user_r - self.global_avg
                self.bias_user[~np.isfinite(self.bias_user)] = 0.0
            else:
                self.bias_user = np.zeros((n_users,))

            if self.include_item:
                self.bias_item = self.bias_item / item_r - self.global_avg
                self.bias_item[~np.isfinite(self.bias_item)] = 0.0
            else:
                self.bias_item = np.zeros((n_items,))

        def predict(self, x):
            inds_u = x[0]
            inds_i = x[1]

            y = np.zeros((len(inds_u,)))

            for ind, (u, i) in enumerate(zip(inds_u, inds_i)):
                y[ind] = self.global_avg + self.bias_user[u] + self.bias_item[i]

            return y

    model = BiasEstimator(include_user, include_item)
    return model


def get_model_ann(meta_users, meta_items):
    lmdba = 0.00007
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
    lmdba = 0.00007
    regularizer = l2(lmdba)

    input_u = Input((1,))
    input_i = Input((1,))

    vec_features_u = Embedding(n_users, n_users_features, input_length=1, trainable=False, name='users_features')(input_u)
    vec_features_u = Flatten()(vec_features_u)

    factors_u = Dense(n_factors, kernel_initializer='normal', activation='linear',
                             kernel_regularizer=regularizer, use_bias=False)(vec_features_u)

    vec_features_i = Embedding(n_items, n_items_feature, input_length=1, trainable=False, name='items_features')(input_i)
    vec_features_i = Flatten()(vec_features_i)

    factors_i = Dense(n_factors, kernel_initializer='normal', activation='linear',
                      kernel_regularizer=regularizer, use_bias=False)(vec_features_i)

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


def get_model_mf_implicit(n_factors, implicit):
    lmdba = 0.00007
    regularizer = l2(lmdba)

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


def get_model_mf_everything(n_factors, n_users_features, n_items_feature):
    lmdba = 0.00007
    regularizer = l2(lmdba)

    input_u = Input((1,))
    input_i = Input((1,))

    # User factors
    vec_features_u = Embedding(n_users, n_users_features, input_length=1, trainable=False, name='users_features')(input_u)
    vec_features_u = Flatten()(vec_features_u)

    factors_u = Dense(n_factors, kernel_initializer='normal', activation='linear',
                             kernel_regularizer=regularizer, use_bias=False)(vec_features_u)

    vec_u = Embedding(n_users, n_factors, input_length=1, embeddings_regularizer=regularizer)(input_u)
    vec_u_r = Flatten()(vec_u)

    vec_implicit = Embedding(n_users, n_items, input_length=1, name='implicit', trainable=False)(input_u)
    implicit_factors = Dense(n_factors, kernel_initializer='normal', activation='linear', kernel_regularizer=regularizer, use_bias=False)(vec_implicit)
    implicit_factors = Flatten()(implicit_factors)

    vec_u_added = Add()([vec_u_r, implicit_factors, factors_u])

    # Item factors
    vec_features_i = Embedding(n_items, n_items_feature, input_length=1, trainable=False, name='items_features')(input_i)
    vec_features_i = Flatten()(vec_features_i)

    factors_i = Dense(n_factors, kernel_initializer='normal', activation='linear',
                      kernel_regularizer=regularizer, use_bias=False)(vec_features_i)

    vec_i = Embedding(n_items, n_factors, input_length=1, embeddings_regularizer=regularizer)(input_i)
    vec_i_r = Flatten()(vec_i)

    vec_i_added = Add()([vec_i_r, factors_i])

    mf = Dot(1)([vec_u_added, vec_i_added])

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

# Apply normalization to features
users_features_norm = users_features / np.sqrt(np.maximum(1, np.sum(users_features, axis=1)[:, None]))
items_features_norm = items_features / np.sqrt(np.maximum(1, np.sum(items_features, axis=1)[:, None]))

# Calculate implicit matrix
implicit = np.zeros((n_users, n_items))

for u, i, r in zip(inds_u_train, inds_i_train, y_train):
    if r >= 0.5:
        implicit[u, i] = 1.0

implicit_norm = implicit / np.sqrt(np.maximum(1, np.sum(implicit, axis=1)[:, None]))

# Build models
models = []

# Bias
# model_bias = get_model_bias()
# models.append(('Bias', model_bias))

# Bias Custom (without Keras)
# model_bias_custom = get_model_bias_custom(include_user=False)
# models.append(('BC_I', model_bias_custom))

# Bias Item only
model_bias_item = get_model_bias_item()
models.append(('BK_I', model_bias_item))

# Ann
model_annf = get_model_ann(users_features_norm, items_features_norm)
models.append(('ANNF', model_annf))

# MF CD
model_mfcd = get_model_mf_cont_demo(20, n_users_features, n_items_features)
model_mfcd.get_layer('users_features').set_weights([users_features_norm])
model_mfcd.get_layer('items_features').set_weights([items_features_norm])
models.append(('MFCD', model_mfcd))

# MF Implicit
# model_mfim = get_model_mf_implicit(20, implicit_norm)
# models.append(('MFIM', model_mfim))

# MF Everything
# model_mfev = get_model_mf_everything(20, n_users_features, n_items_features)
# model_mfev.get_layer('users_features').set_weights([users_features_norm])
# model_mfev.get_layer('items_features').set_weights([items_features_norm])
# model_mfev.get_layer('implicit').set_weights([implicit_norm])
# models.append(('MFEV', model_mfev))

# Training
callbacks = [util.EarlyStoppingBestVal('val_loss', patience=15, min_delta=0.00001)]

for description, model in models:
    model.fit([inds_u_train, inds_i_train], y_train, batch_size=500, epochs=100,
                       validation_split=0.1, verbose=2, callbacks=callbacks)


def test(model, description):
    y = model.predict([inds_u_train, inds_i_train])
    rmse_train = sqrt(mean_squared_error(y_train * 5, y * 5))

    y = model.predict([inds_u_test, inds_i_test])
    rmse_test = sqrt(mean_squared_error(y_test * 5, y * 5))

    print('{}: Training {:.4f}  Testing {:.4f}'.format(description, rmse_train, rmse_test))

print('')
for description, model in models:
    test(model, description)


# Bias: Training 0.9253  Testing 0.9819
# ANN : Training 0.9245  Testing 0.9804
# MFCD: Training 0.9212  Testing 0.9804

