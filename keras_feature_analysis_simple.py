import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import pickle

(X, U, I, Y) = pickle.load(open('data/cont.pickle', 'rb'))

feature_names = ['Age', 'Sex', 'Loc_Lon', 'Loc_Lat', 'job_is_administrator', 'job_is_artist', 'job_is_doctor',
            'job_is_educator', 'job_is_engineer', 'job_is_entertainment', 'job_is_executive', 'job_is_healthcare',
            'job_is_homemaker', 'job_is_lawyer', 'job_is_librarian', 'job_is_marketing', 'job_is_none', 'job_is_other',
            'job_is_programmer', 'job_is_retired', 'job_is_salesman', 'job_is_scientist', 'job_is_student',
            'job_is_technician', 'job_is_writer', 'unknown|0', 'Action|1', 'Adventure|2', 'Animation|3',
            'Children\'s|4', 'Comedy|5', 'Crime|6', 'Documentary|7', 'Drama|8', 'Fantasy|9', 'Film-Noir|10',
            'Horror|11', 'Musical|12', 'Mystery|13', 'Romance|14', 'Sci-Fi|15', 'Thriller|16', 'War|17', 'Western|18']

# Add noise dimension
noisify = True
if noisify:
    noise = np.random.normal(0.0, 0.3, X.shape[0])
    X = np.insert(X, X.shape[1], noise, axis=1)
    feature_names += ['Noise']

n_samples, n_features = X.shape

assert len(feature_names) == n_features

# Normalization
normalize = True
if normalize:
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    # Y -= np.mean(Y)

# Build model
input_x = Input((n_features,))
ann = Dense(1, activation='sigmoid')(input_x)
model = Model(inputs=input_x, outputs=ann)
model.compile(loss='mse', optimizer='adam')

callbacks = [EarlyStopping('val_loss', patience=20)]

model.fit(X, Y / 6, batch_size=500, epochs=100, validation_split=0.1, verbose=2, callbacks=callbacks)

Y_pred = model.predict(X) * 6

rmse = sqrt(mean_squared_error(Y_pred, Y))
print('RMSE training and testing all samples: {}'.format(rmse))

# X_ftest = np.eye(44)
# Y_ftest = model.predict(X_ftest)

# Looking at weights
print('Weights:')
weights_in = np.asarray(model.weights[0].container.data)
bias = np.asarray(model.weights[len(model.weights) - 1].container.data)

print('Bias: {}'.format(bias[0]))
fweights = []
for a in range(n_features):
    weight = weights_in[a][0]
    fweights += [(a, weight)]

fweights = sorted(fweights, key=lambda x: -abs(x[1]))
for fweight in fweights:
    print('Feature {}:  {}'.format(feature_names[fweight[0]], fweight[1]))

# Feature importance testing by setting all inputs of one dimension to its mean (if normalized: 0)
print('\n\nLeaving out one dimension:')
loweights = []
for a in range(n_features):
    X_ftest = X.copy()
    X_ftest[:, a] = 0.0

    Y_pred = model.predict(X_ftest) * 6

    rmse_ftest = sqrt(mean_squared_error(Y_pred, Y))
    loweights += [(a, rmse-rmse_ftest)]

loweights = sorted(loweights, key=lambda x: x[1])
for loweight in loweights:
    print('Feature {}:  {}'.format(feature_names[loweight[0]], loweight[1]))
