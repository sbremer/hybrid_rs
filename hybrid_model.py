import numpy as np
import keras
from keras.layers import Embedding, Reshape, Input, Dense
from keras.layers.merge import Dot, Concatenate, Add
from keras.models import Model
from keras.callbacks import EarlyStopping
from util import EarlyStoppingBestVal
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# Local imports
import util

# Hardcoded config
verbose = 0
batch_size = 2048
val_split = 0.1
bias_mf = False
bias_ann = False


class HybridModel:

    def _get_model_mf(self, n_factors=20, include_bias=False):
        lmdba = 0.00001
        regularizer = keras.regularizers.l2(lmdba)

        input_u = Input((1,))
        input_i = Input((1,))

        vec_u = Embedding(self.n_users, n_factors, input_length=1, embeddings_regularizer=regularizer)(input_u)
        vec_u_r = Reshape((n_factors,))(vec_u)
        vec_i = Embedding(self.n_items, n_factors, input_length=1, embeddings_regularizer=regularizer)(input_i)
        vec_i_r = Reshape((n_factors,))(vec_i)

        mf = Dot(1)([vec_u_r, vec_i_r])

        if include_bias:
            bias_u = Embedding(self.n_users, 1, input_length=1, embeddings_regularizer=regularizer)(input_u)
            bias_u_r = Reshape((1,))(bias_u)
            bias_i = Embedding(self.n_items, 1, input_length=1, embeddings_regularizer=regularizer)(input_i)
            bias_i_r = Reshape((1,))(bias_i)

            added = Add()([bias_u_r, bias_i_r, mf])

            model = Model(inputs=[input_u, input_i], outputs=added)
        else:
            model = Model(inputs=[input_u, input_i], outputs=mf)

        # Compile and return model
        model.compile(loss='mse', optimizer='adamax')
        return model

    def _get_model_ann(self, meta_users, meta_items, include_bias=False):
        lmdba = 0.00001
        regularizer = keras.regularizers.l2(lmdba)

        input_u = Input((1,))
        input_i = Input((1,))

        features_user = meta_users.shape[1]
        features_items = meta_items.shape[1]

        vec_u = Embedding(self.n_users, features_user, input_length=1, trainable=False)(input_u)
        vec_u_r = Reshape((features_user,))(vec_u)
        vec_i = Embedding(self.n_items, features_items, input_length=1, trainable=False)(input_i)
        vec_i_r = Reshape((features_items,))(vec_i)

        vec_features = Concatenate()([vec_u_r, vec_i_r])

        ann_1 = Dense(200, kernel_initializer='uniform', activation='sigmoid')(vec_features)
        ann_2 = Dense(50, kernel_initializer='uniform', activation='sigmoid')(ann_1)
        ann_3 = Dense(1, kernel_initializer='uniform', activation='sigmoid')(ann_2)

        if include_bias:
            bias_u = Embedding(self.n_users, 1, input_length=1, embeddings_regularizer=regularizer)(input_u)
            bias_u_r = Reshape((1,))(bias_u)
            bias_i = Embedding(self.n_items, 1, input_length=1, embeddings_regularizer=regularizer)(input_i)
            bias_i_r = Reshape((1,))(bias_i)

            added = Add()([bias_u_r, bias_i_r, ann_3])

            # output_layer = Dense(1, activation='linear')(added)

            model = Model(inputs=[input_u, input_i], outputs=added)

        else:
            model = Model(inputs=[input_u, input_i], outputs=ann_3)

        model.compile(loss='mse', optimizer='adamax')

        model.layers[2].set_weights([meta_users])
        model.layers[3].set_weights([meta_items])

        return model

    def _prepare_step_data(self, f_xsize, f_tsize, from_mf, shuffle):
        assert self.y_train is not None

        # Get n_xsize random indices not in the training set
        if from_mf:
            inds_u_x, inds_i_x = self.index_gen.get_indices_from_mf(int(self.n_train * f_xsize))
        else:
            inds_u_x, inds_i_x = self.index_gen.get_indices_from_ann(int(self.n_train * f_xsize))

        # Get data for generated indices from other model
        if from_mf:
            y_x = self.model_mf.predict([inds_u_x, inds_i_x]).flatten() + self.mean
        else:
            y_x = self.model_ann.predict([inds_u_x, inds_i_x]).flatten()

        # Take only part of the training dataset
        order = np.arange(self.n_train)
        if f_tsize < 1.0:
            np.random.shuffle(order)
            order = order[:int(self.n_train * f_tsize)]

        # Combine training set and cross-train data
        inds_u_xtrain = np.concatenate((inds_u_x, self.inds_u_train[order]))
        inds_i_xtrain = np.concatenate((inds_i_x, self.inds_i_train[order]))
        y_xtrain = np.concatenate((y_x, self.y_train[order]))

        # Shuffle data
        if shuffle:
            order = np.arange(len(y_xtrain))
            np.random.shuffle(order)
            inds_u_xtrain = inds_u_xtrain[order]
            inds_i_xtrain = inds_i_xtrain[order]
            y_xtrain = y_xtrain[order]

        return inds_u_xtrain, inds_i_xtrain, y_xtrain

    def step_mf(self, f_xsize, f_tsize=1.0, shuffle=True):
        # Get cross-train data from ANN
        inds_u_xtrain, inds_i_xtrain, y_xtrain = self._prepare_step_data(f_xsize, f_tsize, False, shuffle)

        # Update-train MF model with cross-train data
        history = self.model_mf.fit([inds_u_xtrain, inds_i_xtrain], y_xtrain - self.mean, batch_size=batch_size, epochs=100,
                                    validation_split=val_split, verbose=verbose, callbacks=self.callbacks_mf)
        # history = self.model_mf.fit([inds_u_xtrain, inds_i_xtrain], y_xtrain - self.mean, batch_size=batch_size,
        #                             epochs=1, validation_split=val_split, verbose=verbose)

        return history

    def step_ann(self, f_xsize, f_tsize=1.0, shuffle=True):
        # Get cross-train data from MF
        inds_u_xtrain, inds_i_xtrain, y_xtrain = self._prepare_step_data(f_xsize, f_tsize, True, shuffle)

        # Update-train ANN model with cross-train data
        history = self.model_ann.fit([inds_u_xtrain, inds_i_xtrain], y_xtrain, batch_size=batch_size, epochs=100,
                                     validation_split=val_split, verbose=verbose, callbacks=self.callbacks_ann)
        # history = self.model_ann.fit([inds_u_xtrain, inds_i_xtrain], y_xtrain, batch_size=batch_size, epochs=1,
        #                              validation_split=val_split, verbose=verbose)

        return history

    def test_mf(self, inds_u, inds_i, y, prnt=False):
        y_pred = (self.model_mf.predict([inds_u, inds_i]) + self.mean) / 0.2
        rmse = sqrt(mean_squared_error(y / 0.2, y_pred))
        mae = mean_absolute_error(y / 0.2, y_pred)
        if prnt:
            print('RMSE MF: {:.4f} \tMAE: {:.4f}'.format(rmse, mae))
        return rmse

    def test_ann(self, inds_u, inds_i, y, prnt=False):
        y_pred = self.model_ann.predict([inds_u, inds_i]) / 0.2
        rmse = sqrt(mean_squared_error(y / 0.2, y_pred))
        mae = mean_absolute_error(y / 0.2, y_pred)
        if prnt:
            print('RMSE ANN: {:.4f} \tMAE: {:.4f}'.format(rmse, mae))
        return rmse

    def test(self, inds_u_test, inds_i_test, y_test, prnt=False):
        if prnt:
            print('Results of testing:')
        rmse_mf = self.test_mf(inds_u_test, inds_i_test, y_test, prnt)
        rmse_ann = self.test_ann(inds_u_test, inds_i_test, y_test, prnt)
        return rmse_mf, rmse_ann

    def __init__(self, meta_users, meta_items):
        # Get number of users and items
        self.n_users = meta_users.shape[0]
        self.n_items = meta_items.shape[0]

        # Build models
        self.model_mf = self._get_model_mf(n_factors=20, include_bias=bias_mf)
        self.model_ann = self._get_model_ann(meta_users, meta_items, include_bias=bias_ann)

        # Create early stopping callbacks. ANN validation varies more -> higher patience
        self.callbacks_mf = [EarlyStoppingBestVal('val_loss', patience=2)]
        self.callbacks_ann = [EarlyStoppingBestVal('val_loss', patience=4)]

        # Init training set
        self.inds_u_train = None
        self.inds_i_train = None
        self.y_train = None
        self.n_train = 0
        self.index_gen = None

    def train_initial(self, inds_u_train, inds_i_train, y_train, prnt=False):
        # Init training set member variables
        self.inds_u_train = inds_u_train
        self.inds_i_train = inds_i_train
        self.y_train = y_train
        self.n_train = len(y_train)
        self.index_gen = util.IndexGen(self.n_users, self.n_items, inds_u_train, inds_i_train)
        self.mean = np.mean(y_train)

        # Initial early stopping callbacks
        callbacks_mf = [EarlyStoppingBestVal('val_loss', patience=4)]
        callbacks_ann = [EarlyStoppingBestVal('val_loss', patience=8)]

        # callbacks_mf = [EarlyStopping('val_loss', patience=4)]
        # callbacks_ann = [EarlyStopping('val_loss', patience=8)]

        # Run initial training
        history = self.model_mf.fit([inds_u_train, inds_i_train], y_train - self.mean, batch_size=batch_size, epochs=200,
                                    validation_split=val_split, verbose=verbose, callbacks=callbacks_mf)
        history = self.model_ann.fit([inds_u_train, inds_i_train], y_train, batch_size=batch_size, epochs=200,
                                     validation_split=val_split, verbose=verbose, callbacks=callbacks_ann)

        # Print results
        if prnt:
            print('Results of initial training (Training Error):')
        rmse_mf = self.test_mf(inds_u_train, inds_i_train, y_train, prnt)
        rmse_ann = self.test_ann(inds_u_train, inds_i_train, y_train, prnt)

        return rmse_mf, rmse_ann
