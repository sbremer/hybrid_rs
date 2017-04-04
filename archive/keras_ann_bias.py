import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import pickle

import util


# define base mode
def get_model():
    # create model
    ann_input = Input((44,))
    ann_1 = Dense(400, kernel_initializer='uniform', activation='sigmoid')(ann_input)
    ann_2 = Dense(50, kernel_initializer='uniform', activation='sigmoid')(ann_1)
    ann_3 = Dense(1, kernel_initializer='uniform', activation='linear')(ann_2)

    # Compile model
    model = Model(inputs=ann_input, outputs=ann_3)
    model.compile(loss='mse', optimizer='adamax')

    return model

# Hardcoded for now
n_users = 943
n_items = 1682

# fix random seed for reproducibility
seed = 1
np.random.seed(seed)

# evaluate model with standardized dataset
model = get_model()

callbacks = [EarlyStopping('val_loss', patience=5)]

(X, U, I, Y) = pickle.load(open('data/cont.pickle', 'rb'))
U = U.astype(np.int32)
I = I.astype(np.int32)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

n_fold = 5
kf = KFold(n_splits=n_fold, shuffle=True)

rmses_bias = []
rmses_ann = []

# for train_indices, test_indices in kf.split(Y):
for train_indices, test_indices in util.kfold_entries(n_fold, U):
    X_train = X[train_indices, :]
    U_train = U[train_indices]
    I_train = I[train_indices]
    Y_train = Y[train_indices]

    X_test = X[test_indices, :]
    U_test = U[test_indices]
    I_test = I[test_indices]
    Y_test = Y[test_indices]

    bias_global = 0.0
    bias_global_n = 0
    bias_u = np.zeros(n_users)
    bias_u_n = np.zeros(n_users)
    bias_i = np.zeros(n_items)
    bias_i_n = np.zeros(n_items)

    for i in range(len(train_indices)):
        user = U_train[i]
        item = I_train[i]
        rating = Y_train[i]

        bias_global += rating
        bias_global_n += 1
        bias_u[user] += rating
        bias_u_n[user] += 1
        bias_i[item] += rating
        bias_i_n[item] += 1

    bias_global /= bias_global_n
    bias_u = bias_u / bias_u_n - bias_global
    bias_i = bias_i / bias_i_n - bias_global

    # Make sure no Nans are in the bias arrays
    bias_u[np.isnan(bias_u)] = 0.0
    bias_i[np.isnan(bias_i)] = 0.0

    Y_train_bias = np.zeros_like(Y_train)

    for i in range(len(train_indices)):
        user = U_train[i]
        item = I_train[i]
        rating = Y_train[i]

        Y_train_bias[i] = bias_global + bias_u[user] + bias_i[item]

    Y_train -= Y_train_bias

    model = get_model()
    history = model.fit(X_train, Y_train, batch_size=500, epochs=100, validation_split=0.2, verbose=2, callbacks=callbacks)

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    Y_test_bias = np.zeros_like(Y_test)

    for i in range(len(test_indices)):
        user = U_test[i]
        item = I_test[i]
        rating = Y_test[i]

        Y_test_bias[i] = bias_global + bias_u[user] + bias_i[item]

    rmse_bias = sqrt(mean_squared_error(Y_test_bias, Y_test))
    print('RMSE Bias only (Baseline): {}'.format(rmse_bias))
    rmses_bias.append(rmse_bias)

    Y_pred = model.predict(X_test).flatten() + Y_test_bias

    rmse_ann = sqrt(mean_squared_error(Y_pred, Y_test))
    print('RMSE Ann: {}'.format(rmse_ann))
    rmses_ann.append(rmse_ann)

print('Crossval RMSE Bias only (Baseline): {}'.format(np.mean(rmses_bias)))
print('Crossval RMSE Ann: {}'.format(np.mean(rmses_ann)))

# kfold = KFold(n_splits=3, random_state=seed)
# results = cross_val_score(model, X, Y, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
