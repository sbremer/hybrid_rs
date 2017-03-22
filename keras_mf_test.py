import numpy as np
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.models import Sequential
import pickle
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import KFold

# Hardcoded for now
n_users = 943
n_items = 1682

k_factors = 100

model = Sequential()
P = Sequential()
P.add(Embedding(n_users, k_factors, input_length=1))
P.add(Reshape((k_factors,)))
Q = Sequential()
Q.add(Embedding(n_items, k_factors, input_length=1))
Q.add(Reshape((k_factors,)))
model.add(Merge([P, Q], mode='dot', dot_axes=1))
model.compile(loss='mse', optimizer='adamax')

(X, U, I, Y) = pickle.load(open('data/cont.pickle', 'rb'))

callbacks = [EarlyStopping('val_loss', patience=4)]


n_fold = 5
kf = KFold(n_splits=n_fold, shuffle=True)

rmses = []

for train_indices, test_indices in kf.split(Y):
    U_train = U[train_indices]
    I_train = I[train_indices]
    Y_train = Y[train_indices] / 6

    U_test = U[test_indices]
    I_test = I[test_indices]
    Y_test = Y[test_indices]

    history = model.fit([U_train, I_train], Y_train, epochs=30, validation_split=0.1, verbose=2, callbacks=callbacks)

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    Y_pred = model.predict([U_test, I_test]) * 6

    rmse = sqrt(mean_squared_error(Y_pred, Y_test))
    print('RMSE: {}'.format(rmse))
    rmses.append(rmse)

print('Crossval RMSE of Content-based RS: {}'.format(np.mean(rmses)))


