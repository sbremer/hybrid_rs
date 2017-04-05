import numpy as np
import keras
from keras.layers import Embedding, Reshape, Input, Dense
from keras.layers.merge import Dot, Concatenate, Add
from keras.models import Model
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle

# Local imports
import util


def _get_magic_model(meta_users, meta_items, n_factors=20):
    lmdba = 0.00001
    regularizer = keras.regularizers.l2(lmdba)

    input_u = Input((1,))
    input_i = Input((1,))

    n_users, features_user = meta_users.shape[:2]
    n_items, features_items = meta_items.shape[:2]

    vec_u = Embedding(n_users, n_factors, input_length=1, embeddings_regularizer=regularizer)(input_u)
    vec_u_r = Reshape((n_factors,))(vec_u)
    vec_i = Embedding(n_items, n_factors, input_length=1, embeddings_regularizer=regularizer)(input_i)
    vec_i_r = Reshape((n_factors,))(vec_i)

    mf = Dot(1)([vec_u_r, vec_i_r])

    bias_u = Embedding(n_users, 1, input_length=1, embeddings_regularizer=regularizer)(input_u)
    bias_u_r = Reshape((1,))(bias_u)
    bias_i = Embedding(n_items, 1, input_length=1, embeddings_regularizer=regularizer)(input_i)
    bias_i_r = Reshape((1,))(bias_i)

    added_mf = Add()([bias_u_r, bias_i_r, mf])

    vec_feature_u = Embedding(n_users, features_user, input_length=1, trainable=False)(input_u)
    vec_feature_u_r = Reshape((features_user,))(vec_feature_u)
    vec_feature_i = Embedding(n_items, features_items, input_length=1, trainable=False)(input_i)
    vec_feature_i_r = Reshape((features_items,))(vec_feature_i)

    vec_features = Concatenate()([vec_u_r, vec_i_r, vec_feature_u_r, vec_feature_i_r])

    ann_1 = Dense(200, kernel_initializer='uniform', activation='sigmoid')(vec_features)
    ann_2 = Dense(50, kernel_initializer='uniform', activation='sigmoid')(ann_1)
    ann_3 = Dense(1, kernel_initializer='uniform', activation='sigmoid')(ann_2)

    # added_ann = Add()([bias_u_r, bias_i_r, ann_3])

    model_mf = Model(inputs=[input_u, input_i], outputs=added_mf)
    model_ann = Model(inputs=[input_u, input_i], outputs=ann_3)

    # Compile and return model
    model_mf.compile(loss='mse', optimizer='adamax')
    model_ann.compile(loss='mse', optimizer='adam')

    model_ann.layers[4].set_weights([meta_users])
    model_ann.layers[5].set_weights([meta_items])

    return model_mf, model_ann


(meta_users, meta_items) = pickle.load(open('data/imdb_metadata.pickle', 'rb'))
(_, inds_u, inds_i, y) = pickle.load(open('data/cont.pickle', 'rb'))

# meta_items = meta_items[:, 19:21]

# Normalize features and set Nans to zero (=mean)
meta_users = (meta_users - np.nanmean(meta_users, axis=0)) / np.nanstd(meta_users, axis=0)
meta_items = (meta_items - np.nanmean(meta_items, axis=0)) / np.nanstd(meta_items, axis=0)
meta_users[np.isnan(meta_users)] = 0
meta_items[np.isnan(meta_items)] = 0

# Rescale ratings to ~(0.0, 1.0)
y_org = y.copy()
y = (y - 0.5) * 0.2

model_mf, model_ann = _get_magic_model(meta_users, meta_items)

# Crossvalidation
n_fold = 5
user_coldstart = True
if user_coldstart:
    kfold = util.kfold_entries(n_fold, inds_u)
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

callbacks = [EarlyStopping('val_loss', patience=4)]


history = model_mf.fit([inds_u_train, inds_i_train], y_train, batch_size=500, epochs=100,
                                     validation_split=0.2, verbose=2, callbacks=callbacks)

y_pred_mf = model_mf.predict([inds_u_test, inds_i_test])/ 0.2 + 0.5
rmse_mf = sqrt(mean_squared_error(y_test / 0.2 + 0.5, y_pred_mf))
print('RMSE MF: {0:.4f}'.format(rmse_mf))


assert (np.array(model_mf.layers[2].get_weights()) == np.array(model_ann.layers[2].get_weights())).all()
assert (np.array(model_mf.layers[3].get_weights()) == np.array(model_ann.layers[3].get_weights())).all()

model_ann.layers[2].trainable = False
model_ann.layers[3].trainable = False
model_ann.layers[12].trainable = False
model_ann.layers[13].trainable = False

history = model_ann.fit([inds_u_train, inds_i_train], y_train, batch_size=500, epochs=100,
                                     validation_split=0.2, verbose=2, callbacks=callbacks)

y_pred_ann = model_ann.predict([inds_u_test, inds_i_test])/ 0.2 + 0.5
rmse_ann = sqrt(mean_squared_error(y_test / 0.2 + 0.5, y_pred_ann))
print('RMSE ANN: {0:.4f}'.format(rmse_ann))
