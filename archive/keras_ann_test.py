import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import pickle


# define base mode
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(200, input_dim=44, kernel_initializer='uniform', activation='tanh'))
    model.add(Dense(50, kernel_initializer='uniform', activation='tanh'))
    model.add(Dense(1, kernel_initializer='uniform', activation='linear'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Hardcoded for now
n_users = 943
n_items = 1682

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
model = KerasRegressor(build_fn=baseline_model, nb_epoch=10, batch_size=100, verbose=2)

(X, U, I, Y) = pickle.load(open('data/cont.pickle', 'rb'))
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

n_fold = 2
kf = KFold(n_splits=n_fold, shuffle=True)

rmses = []

for train_indices, test_indices in kf.split(Y):
    X_train = X[train_indices, :]
    U_train = U[train_indices]
    I_train = I[train_indices]
    Y_train = Y[train_indices] / 6

    X_test = X[test_indices, :]
    U_test = U[test_indices]
    I_test = I[test_indices]
    Y_test = Y[test_indices]

    history = model.fit(X_train, Y_train, validation_split=0.33, epochs=50)

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    Y_pred = model.predict(X_test) * 6

    rmse = sqrt(mean_squared_error(Y_pred, Y_test))
    print('RMSE: {}'.format(rmse))
    rmses.append(rmse)

print('Crossval RMSE of Content-based RS: {}'.format(np.mean(rmses)))

# kfold = KFold(n_splits=3, random_state=seed)
# results = cross_val_score(model, X, Y, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
