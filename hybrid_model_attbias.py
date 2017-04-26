import numpy as np

# Keras
from keras.layers import Embedding,  Input, Dense, Flatten
from keras.layers.merge import Dot, Concatenate, Add, Multiply
from keras.regularizers import l2
from keras.models import Model
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# Local imports
import util

# Hardcoded config
verbose = 0

batch_size_init = 512
batch_size_xtrain = 1024

val_split_init = 0.1
val_split_xtrain = 0.2

implicit_thresh_init = 0.4
implicit_thresh_xtrain = 0.7

optimizer_init = 'nadam'
optimizer_xtrain = 'adadelta'


class HybridModel:

    def __init__(self, meta_users, meta_items):
        # Get number of users and items
        self.n_users = meta_users.shape[0]
        self.n_items = meta_items.shape[0]

        # Create early stopping callbacks. ANN validation varies more -> higher patience
        self.callbacks_mf = [util.EarlyStoppingBestVal('val_loss', patience=2)]
        self.callbacks_ann = [util.EarlyStoppingBestVal('val_loss', patience=4)]

        # Build models
        self.model_mf = self._get_model_mf(40)
        self.model_cs = self._get_model_cs(meta_users, meta_items)

        # Init training set
        self.inds_u_train = None
        self.inds_i_train = None
        self.y_train = None
        self.n_train = 0
        self.index_gen = None

    def _get_model_mf(self, n_factors):
        lmdba = 0.00005
        regularizer = l2(lmdba)

        input_u = Input((1,))
        input_i = Input((1,))

        vec_u = Embedding(self.n_users, n_factors, input_length=1, embeddings_regularizer=regularizer)(input_u)
        vec_u_r = Flatten()(vec_u)
        vec_i = Embedding(self.n_items, n_factors, input_length=1, embeddings_regularizer=regularizer)(input_i)
        vec_i_r = Flatten()(vec_i)

        vec_implicit = Embedding(self.n_users, self.n_items, input_length=1, trainable=False, name='implicit')(input_u)
        implicit_factors = Dense(n_factors, kernel_initializer='normal', activation='linear',
                                 kernel_regularizer=regularizer)(vec_implicit)
        implicit_factors = Flatten()(implicit_factors)

        vec_u_added = Add()([vec_u_r, implicit_factors])

        mf = Dot(1)([vec_u_added, vec_i_r])

        bias_u = Embedding(self.n_users, 1, input_length=1, embeddings_regularizer=regularizer)(input_u)
        bias_u_r = Flatten()(bias_u)
        bias_i = Embedding(self.n_items, 1, input_length=1, embeddings_regularizer=regularizer)(input_i)
        bias_i_r = Flatten()(bias_i)

        added = Concatenate()([bias_u_r, bias_i_r, mf])

        mf_out = util.BiasLayer()(added)

        model = Model(inputs=[input_u, input_i], outputs=mf_out)

        # model.layers[1].set_weights([implicit])

        # Compile and return model
        model.compile(loss='mse', optimizer=optimizer_init)
        return model

    def _get_model_cs(self, meta_users, meta_items):
        lmdba = 0.00003
        regularizer = l2(lmdba)

        input_u = Input((1,))
        input_i = Input((1,))

        n_users_features = meta_users.shape[1]
        n_items_feature = meta_items.shape[1]

        vec_features_u = Embedding(self.n_users, n_users_features, input_length=1, trainable=False,
                                   name='users_features')(input_u)
        vec_features_u = Flatten()(vec_features_u)

        vec_features_i = Embedding(self.n_items, n_items_feature, input_length=1, trainable=False,
                                   name='items_features')(input_i)
        vec_features_i = Flatten()(vec_features_i)

        factors_i = Dense(n_users_features, kernel_initializer='normal', activation='linear',
                          kernel_regularizer=l2(0.002), use_bias=False)(vec_features_i)

        # 0.002: 1.0339

        mult = Multiply()([factors_i, vec_features_u])

        bias_u = Embedding(self.n_users, 1, input_length=1, embeddings_initializer='zeros',
                           embeddings_regularizer=regularizer)(input_u)
        bias_u = Flatten()(bias_u)
        bias_i = Embedding(self.n_items, 1, input_length=1, embeddings_initializer='zeros',
                           embeddings_regularizer=regularizer)(input_i)
        bias_i = Flatten()(bias_i)

        concat = Concatenate()([bias_u, bias_i, mult])

        cs_out = Dense(1, activation='linear', use_bias=True)(concat)

        model = Model(inputs=[input_u, input_i], outputs=cs_out)

        # Compile and return model
        model.compile(loss='mse', optimizer=optimizer_init)

        # Normalize Genre matrix and set static weights
        meta_items = meta_items / np.maximum(1, np.sum(meta_items, axis=1)[:, None])
        model.get_layer('users_features').set_weights([meta_users])
        model.get_layer('items_features').set_weights([meta_items])

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
            y_x = self.model_mf.predict([inds_u_x, inds_i_x]).flatten()
        else:
            y_x = self.model_cs.predict([inds_u_x, inds_i_x]).flatten()

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

    def _recompute_implicit(self, inds_u, inds_i, y, init=False):

        if init:
            thresh = implicit_thresh_init
        else:
            thresh = implicit_thresh_xtrain

        # Use ratings over the threshold as implicit feedback
        for u, i, r in zip(inds_u, inds_i, y):
            if r >= thresh:
                self.implicit[u, i] = 1.0

        # Normalize using sqrt (ref. SVD++ paper)
        implicit_norm = self.implicit / np.sqrt(np.maximum(1, np.sum(self.implicit, axis=1)[:, None]))

        self.model_mf.get_layer('implicit').set_weights([implicit_norm])

    def step_mf(self, f_xsize, f_tsize=1.0, shuffle=True):
        # Get cross-train data from ANN
        inds_u_xtrain, inds_i_xtrain, y_xtrain = self._prepare_step_data(f_xsize, f_tsize, False, shuffle)

        # Recompute implicit matrix
        self._recompute_implicit(inds_u_xtrain, inds_i_xtrain, y_xtrain)

        # Update-train MF model with cross-train data
        history = self.model_mf.fit([inds_u_xtrain, inds_i_xtrain], y_xtrain, batch_size=batch_size_xtrain, epochs=150,
                                    validation_split=val_split_xtrain, verbose=verbose, callbacks=self.callbacks_mf)
        # history = self.model_mf.fit([inds_u_xtrain, inds_i_xtrain], y_xtrain - self.mean, batch_size=batch_size,
        #                             epochs=1, validation_split=val_split, verbose=verbose)

        return history

    def step_cs(self, f_xsize, f_tsize=1.0, shuffle=True):
        # Get cross-train data from MF
        inds_u_xtrain, inds_i_xtrain, y_xtrain = self._prepare_step_data(f_xsize, f_tsize, True, shuffle)

        # Update-train ANN model with cross-train data
        history = self.model_cs.fit([inds_u_xtrain, inds_i_xtrain], y_xtrain, batch_size=batch_size_xtrain, epochs=150,
                                    validation_split=val_split_xtrain, verbose=verbose, callbacks=self.callbacks_ann)
        # history = self.model_ann.fit([inds_u_xtrain, inds_i_xtrain], y_xtrain, batch_size=batch_size, epochs=1,
        #                              validation_split=val_split, verbose=verbose)

        return history

    def test_mf(self, inds_u, inds_i, y, prnt=False):
        y_pred = self.model_mf.predict([inds_u, inds_i]) / 0.2
        rmse = sqrt(mean_squared_error(y / 0.2, y_pred))
        mae = mean_absolute_error(y / 0.2, y_pred)
        if prnt:
            print('RMSE MF: {:.4f} \tMAE: {:.4f}'.format(rmse, mae))
        return rmse

    def test_cs(self, inds_u, inds_i, y, prnt=False):
        y_pred = self.model_cs.predict([inds_u, inds_i]) / 0.2
        rmse = sqrt(mean_squared_error(y / 0.2, y_pred))
        mae = mean_absolute_error(y / 0.2, y_pred)
        if prnt:
            print('RMSE ANN: {:.4f} \tMAE: {:.4f}'.format(rmse, mae))
        return rmse

    def test(self, inds_u_test, inds_i_test, y_test, prnt=False):
        if prnt:
            print('Results of testing:')
        rmse_mf = self.test_mf(inds_u_test, inds_i_test, y_test, prnt)
        rmse_cs = self.test_cs(inds_u_test, inds_i_test, y_test, prnt)
        return rmse_mf, rmse_cs

    def train_initial(self, inds_u_train, inds_i_train, y_train, prnt=False):
        # Init training set member variables
        self.inds_u_train = inds_u_train
        self.inds_i_train = inds_i_train
        self.y_train = y_train
        self.n_train = len(y_train)
        self.index_gen = IndexGen(self.n_users, self.n_items, inds_u_train, inds_i_train)

        self.implicit = np.zeros((self.n_users, self.n_items))

        self._recompute_implicit(inds_u_train, inds_i_train, y_train, True)

        # Initial early stopping callbacks
        callbacks_mf = [util.EarlyStoppingBestVal('val_loss', patience=10)]
        callbacks_cs = [util.EarlyStoppingBestVal('val_loss', patience=10)]

        # callbacks_mf = [EarlyStopping('val_loss', patience=4)]
        # callbacks_cs = [EarlyStopping('val_loss', patience=8)]

        # Run initial training
        history = self.model_mf.fit([inds_u_train, inds_i_train], y_train, batch_size=batch_size_init, epochs=200,
                                    validation_split=val_split_init, verbose=verbose, callbacks=callbacks_mf)
        history = self.model_cs.fit([inds_u_train, inds_i_train], y_train, batch_size=batch_size_init, epochs=200,
                                    validation_split=val_split_init, verbose=verbose, callbacks=callbacks_cs)

        # Recompile model with other optimizer for continued cross training
        self.model_mf.compile(loss='mse', optimizer=optimizer_xtrain)
        self.model_cs.compile(loss='mse', optimizer=optimizer_xtrain)

        # Print results
        if prnt:
            print('Results of initial training (Training Error):')
        rmse_mf = self.test_mf(inds_u_train, inds_i_train, y_train, prnt)
        rmse_cs = self.test_cs(inds_u_train, inds_i_train, y_train, prnt)

        return rmse_mf, rmse_cs

    def xtraining_complete(self, test_data=None):

        # Config for early stopping
        patience = 4
        min_delta = 0.00005

        vloss_mf_min = float('inf')
        epoch_min = 0

        # Alternating cross-training
        for i in range(20):
            print('Training step {}'.format(i + 1))

            # ANN step
            history_cs = self.step_cs(0.2, 1.0, True)
            vloss_cs = min(history_cs.history['val_loss'])

            # MF step
            history_mf = self.step_mf(0.2, 1.0, True)
            vloss_mf = min(history_mf.history['val_loss'])

            if test_data:
                print('Results after training step {}:'.format(i + 1))
                inds_u_test = test_data[0]
                inds_i_test = test_data[1]
                y_test = test_data[2]
                rmse_mf, rmse_cs = self.test(inds_u_test, inds_i_test, y_test, True)

            # Check for early stopping
            if vloss_mf < vloss_mf_min - min_delta:
                vloss_mf_min = vloss_mf
                epoch_min = i
            else:
                if i > epoch_min + patience:
                    print('Stopping crosstraining after epoch {}'.format(i+1))
                    break


class IndexGen:
    """
    Generates indices for user-item pairs which are not in the training set.
    Build for sparse training set! Might be very inefficient else.
    """

    def __init__(self, n_users, n_items, inds_u, inds_i):
        self.n_users = n_users
        self.n_items = n_items

        # Create probability distributions for getting data
        user_counts = np.bincount(inds_u, minlength=n_users)
        item_counts = np.bincount(inds_i, minlength=n_items)

        # user_counts = np.ones((n_users,))
        # item_counts = np.ones((n_items,))

        self.prob_from_mf_user = user_counts / np.sum(user_counts)
        self.prob_from_mf_item = item_counts / np.sum(item_counts)

        user_counts_adj = np.maximum(4, user_counts) ** 2
        item_counts_adj = np.maximum(4, item_counts) ** 2

        item_counts_adj = np.ones((n_items,))

        self.prob_from_ann_user = (1 / user_counts_adj) / np.sum(1 / user_counts_adj)
        self.prob_from_ann_item = (1 / item_counts_adj) / np.sum(1 / item_counts_adj)

        # Create and fill entry lookup table
        self.lookup = {}
        for u_i in zip(inds_u, inds_i):
            self.lookup[u_i] = True

    def get_indices(self, n_indices):
        inds_u = np.zeros((n_indices,), np.int)
        inds_i = np.zeros((n_indices,), np.int)

        lookup_samples = {}

        got = 0
        while got < n_indices:
            u = np.random.randint(self.n_users)
            i = np.random.randint(self.n_items)

            if (u, i) not in self.lookup and (u, i) not in lookup_samples:
                inds_u[got] = u
                inds_i[got] = i
                lookup_samples[(u, i)] = True
                got += 1

        return inds_u, inds_i

    def get_indices_from_mf(self, n_indices):
        inds_u = np.zeros((n_indices,), np.int)
        inds_i = np.zeros((n_indices,), np.int)

        lookup_samples = {}

        got = 0
        while got < n_indices:
            u = np.random.choice(np.arange(self.n_users), p=self.prob_from_mf_user)
            i = np.random.choice(np.arange(self.n_items), p=self.prob_from_mf_item)

            if (u, i) not in self.lookup and (u, i) not in lookup_samples:
                inds_u[got] = u
                inds_i[got] = i
                lookup_samples[(u, i)] = True
                got += 1

        return inds_u, inds_i

    def get_indices_from_ann(self, n_indices):
        inds_u = np.zeros((n_indices,), np.int)
        inds_i = np.zeros((n_indices,), np.int)

        lookup_samples = {}

        got = 0
        while got < n_indices:
            u = np.random.choice(np.arange(self.n_users), p=self.prob_from_ann_user)
            i = np.random.choice(np.arange(self.n_items), p=self.prob_from_ann_item)

            if (u, i) not in self.lookup and (u, i) not in lookup_samples:
                inds_u[got] = u
                inds_i[got] = i
                lookup_samples[(u, i)] = True
                got += 1

        return inds_u, inds_i