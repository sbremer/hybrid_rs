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


def get_model():
    # Config
    n_factors = 20
    lmdba = 0.00001
    regularizer = keras.regularizers.l2(lmdba)

    input_u = Input((1,))
    input_i = Input((1,))

    vec_u = Embedding(n_users, n_factors, input_length=1, embeddings_regularizer=regularizer)(input_u)
    vec_u = Reshape((n_factors,))(vec_u)
    vec_i = Embedding(n_items, n_factors, input_length=1, embeddings_regularizer=regularizer)(input_i)
    vec_i = Reshape((n_factors,))(vec_i)

    mf = Dot(1)([vec_u, vec_i])

    bias_u = Embedding(n_users, 1, input_length=1, embeddings_regularizer=regularizer)(input_u)
    bias_u = Reshape((1,))(bias_u)
    bias_i = Embedding(n_items, 1, input_length=1, embeddings_regularizer=regularizer)(input_i)
    bias_i = Reshape((1,))(bias_i)

    input_x = Input((44,))
    ann_1 = Dense(200, input_dim=44, kernel_initializer='uniform', activation='sigmoid')(input_x)
    ann_2 = Dense(50, kernel_initializer='uniform', activation='sigmoid')(ann_1)
    ann_3 = Dense(1, kernel_initializer='uniform', activation='sigmoid')(ann_2)

    # added = Add()([mf, bias_u, bias_i, ann_3])
    added = Add()([bias_u, bias_i, ann_3])

    model = Model(inputs=[input_u, input_i, input_x], outputs=bias_i)

    model.compile(loss='mse', optimizer='adamax')
    return model

(X, U, I, Y) = pickle.load(open('data/cont.pickle', 'rb'))

# Normalize X
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Y -= np.mean(Y)
Y = (Y - 0.5) * 0.2

callbacks = [EarlyStopping('val_loss', patience=4)]

# Init random seed for reproducibility
# seed = 7
# np.random.seed(seed)

rmses = []

# Crossvalidation
n_fold = 5
coldstart = True
if coldstart:
    kfold = util.kfold_entries(n_fold, U)
else:
    kfold = util.kfold(n_fold, U)

for train_indices, test_indices in kfold:
    X_train = X[train_indices, :]
    U_train = U[train_indices]
    I_train = I[train_indices]
    Y_train = Y[train_indices]

    X_test = X[test_indices, :]
    U_test = U[test_indices]
    I_test = I[test_indices]
    Y_test = Y[test_indices]

    model = get_model()
    history = model.fit([U_train, I_train, X_train], Y_train, batch_size=500, epochs=100, validation_split=0.1, verbose=2, callbacks=callbacks)

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    Y_pred = model.predict([U_test, I_test, X_test]) / 0.2 + 0.5

    rmse = sqrt(mean_squared_error(Y_pred, Y_test / 0.2 + 0.5))
    print('RMSE: {}'.format(rmse))
    rmses.append(rmse)

print('Crossval RMSE of MF based RS: {0:.4f}'.format(np.mean(rmses)))


