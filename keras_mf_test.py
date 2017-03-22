import numpy as np
from keras.layers import Embedding, Reshape, Input, Dense
from keras.layers.merge import Dot, Concatenate, Add
from keras.models import Model
from keras.callbacks import EarlyStopping

import keras
import pickle
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import KFold


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

    all = Add()([mf, bias_u, bias_i])

    model = Model(inputs=[input_u, input_i], outputs=all)

    model.compile(loss='mse', optimizer='adamax')
    return model

(X, U, I, Y) = pickle.load(open('data/cont.pickle', 'rb'))

Y -= np.mean(Y)

callbacks = [EarlyStopping('val_loss', patience=4)]


n_fold = 5
kf = KFold(n_splits=n_fold, shuffle=True)

rmses = []

for train_indices, test_indices in kf.split(Y):
    U_train = U[train_indices]
    I_train = I[train_indices]
    Y_train = Y[train_indices] / 5

    U_test = U[test_indices]
    I_test = I[test_indices]
    Y_test = Y[test_indices]

    model = get_model()
    history = model.fit([U_train, I_train], Y_train, batch_size=500, epochs=100, validation_split=0.1, verbose=2, callbacks=callbacks)

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    Y_pred = model.predict([U_test, I_test]) * 5

    rmse = sqrt(mean_squared_error(Y_pred, Y_test))
    print('RMSE: {}'.format(rmse))
    rmses.append(rmse)

print('Crossval RMSE of MF based RS: {}'.format(np.mean(rmses)))


