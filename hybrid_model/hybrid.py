from typing import NewType, NamedTuple, List, Type, Dict

import numpy as np
from keras.optimizers import Optimizer
from keras.callbacks import EarlyStopping

from hybrid_model.index_sampler import IndexSampler
from hybrid_model.models import AbstractModelCF, AbstractModelMD
from hybrid_model.transform import Transformation
from util.callbacks_custom import EarlyStoppingBestVal

Matrix = NewType('Matrix', np.ndarray)


class HybridConfig(NamedTuple):
    """
    Config Class for the Hybrid model
    """
    model_type_cf: Type[AbstractModelCF]
    model_config_cf: Dict
    model_type_md: Type[AbstractModelMD]
    model_config_md: Dict

    opt_cf_init: Optimizer
    opt_md_init: Optimizer
    opt_cf_xtrain: Optimizer
    opt_md_xtrain: Optimizer

    batch_size_init_cf: int
    batch_size_init_md: int
    batch_size_xtrain_cf: int
    batch_size_xtrain_md: int
    val_split_init: float
    val_split_xtrain: float

    index_sampler: Type[IndexSampler]
    index_sampler_config: Dict

    xtrain_epochs: int
    xtrain_data_shuffle: bool

    transformation: Transformation


class HybridModel:
    """
    Hybrid model
    Training the matrix factorization model as well as the cold start model.
    """

    def __init__(self, meta_users: Matrix, meta_items: Matrix, config: HybridConfig, verbose: int = 0):
        self.config = config
        self.verbose = verbose

        # Get number of users and items
        self.n_users = meta_users.shape[0]
        self.n_items = meta_items.shape[0]

        # Prepare config for building models
        type_cf = config.model_type_cf
        config_cf = config.model_config_cf
        type_md = config.model_type_md
        config_md = config.model_config_md
        transformation = config.transformation

        # Build models
        self.model_cf = type_cf(self.n_users, self.n_items, config_cf, transformation)
        self.model_md = type_md(meta_users, meta_items, config_md, transformation)

        # Callbacks for early stopping during one cross-trainin iteration
        self.callbacks_cf = [EarlyStopping('val_loss', patience=0)]
        self.callbacks_md = [EarlyStopping('val_loss', patience=0)]

        # Init to be training data
        self.x_train: List[Matrix] = None
        self.y_train: Matrix = None
        self.n_train: int = 0

        self.user_dist: Matrix = None
        self.item_dist: Matrix = None

        self.index_sampler: IndexSampler = None

    def fit(self, x_train, y_train):

        # Initial training
        self.fit_init(x_train, y_train)

        # Cross training
        self.fit_cross()

    def fit_init(self, x_train, y_train):

        self.setup_init_training()

        # Transform y data
        y_train = self.config.transformation.transform(y_train)

        # Reshape data into "unflattened" arrays, Keras will do that anyway
        self.x_train = [x_train[0].reshape((len(x_train[0]), 1)), x_train[1].reshape((len(x_train[1]), 1))]
        self.n_train = len(y_train)
        self.y_train = y_train.reshape((self.n_train, 1))

        # Get distribution of ratings per user and item
        self.user_dist = np.bincount(x_train[0], minlength=self.n_users)
        self.item_dist = np.bincount(x_train[1], minlength=self.n_items)

        # Initially train models separately
        self._train_init()

    def setup_init_training(self):
        pass
        # Compile model using optimizer used for initial training
        self.model_cf.compile(self.config.opt_cf_init)
        self.model_md.compile(self.config.opt_md_init)

    def _train_init(self):

        # Initial early stopping callbacks
        callbacks_cf = [EarlyStoppingBestVal('val_loss', patience=5)]
        callbacks_md = [EarlyStoppingBestVal('val_loss', patience=5)]

        # Compute implicit matrix for matrix factorization
        self.model_cf.recompute_implicit(self.x_train, self.y_train, transformed=True)

        # Train both models with the training data only
        self.model_cf.model.fit(self.x_train, self.y_train, batch_size=self.config.batch_size_init_cf, epochs=200,
                                validation_split=self.config.val_split_init, verbose=self.verbose, callbacks=callbacks_cf)
        self.model_md.model.fit(self.x_train, self.y_train, batch_size=self.config.batch_size_init_md, epochs=200,
                                validation_split=self.config.val_split_init, verbose=self.verbose, callbacks=callbacks_md)

    def fit_cross(self):
        self.setup_cross_training()

        # Alternating cross-training for a fixed number of epochs
        for i in range(self.config.xtrain_epochs):

            if self.verbose != 0:
                print('Training step {}'.format(i + 1))

            self.fit_cross_epoch()

    def setup_cross_training(self):

        # Init Index Sampler
        self.index_sampler = self.config.index_sampler(self.user_dist, self.item_dist,
                                                       self.config.index_sampler_config, self.x_train)

        # Recompile model using optimizer for cross-training
        # self.model_cf.compile(self.config.opt_cf_xtrain)
        # self.model_md.compile(self.config.opt_md_xtrain)

    def fit_cross_epoch(self):

        # Get data from MD to train CF
        self._step_md_cf()

        # Vice versa
        self._step_cf_md()

    def _step_md_cf(self):

        # Get indices for cross training from sampling distribution
        inds_u_x, inds_i_x = self.index_sampler.get_indices_from_md()

        # Get prediction on sampled indices
        y_x = self.model_md.model.predict([inds_u_x, inds_i_x])

        # Recompute implicit matrix
        self.model_cf.recompute_implicit([inds_u_x, inds_i_x], y_x, transformed=True, crosstrain=True)

        # Combine data with original training data
        inds_u_xtrain, inds_i_xtrain, y_xtrain = self._concat_data(inds_u_x, inds_i_x, y_x, self.config.xtrain_data_shuffle)

        # Update-train CF model with cross-train data
        history = self.model_cf.model.fit([inds_u_xtrain, inds_i_xtrain], y_xtrain, batch_size=self.config.batch_size_xtrain_cf,
                                          epochs=150, validation_split=self.config.val_split_xtrain, verbose=self.verbose,
                                          callbacks=self.callbacks_cf)

        # Return best validation loss
        return min(history.history['val_loss'])

    def _step_cf_md(self):

        # Get indices for cross training from sampling distribution
        inds_u_x, inds_i_x = self.index_sampler.get_indices_from_cf()

        # Get prediction on sampled indices
        y_x = self.model_cf.model.predict([inds_u_x, inds_i_x])

        # Combine data with original training data
        inds_u_xtrain, inds_i_xtrain, y_xtrain = self._concat_data(inds_u_x, inds_i_x, y_x, self.config.xtrain_data_shuffle)

        # Update-train MD model with cross-train data
        history = self.model_md.model.fit([inds_u_xtrain, inds_i_xtrain], y_xtrain, batch_size=self.config.batch_size_xtrain_md,
                                          epochs=150, validation_split=self.config.val_split_xtrain, verbose=self.verbose,
                                          callbacks=self.callbacks_md)

        # Return best validation loss
        return min(history.history['val_loss'])

    # def test_cf(self, x_test, y_test, prnt=False):
    #     y_pred = self.model_cf.predict(x_test)
    #
    #     result = evaluation.EvaluationResultPart()
    #     for measure, metric in evaluation.metrics_rmse.items():
    #         result.results[measure] = metric.calculate(y_test, y_pred, x_test)
    #
    #     if prnt:
    #         print('CF: ', result)
    #
    #     return result
    #
    # def test_md(self, x_test, y_test, prnt=False):
    #     y_pred = self.model_md.predict(x_test)
    #
    #     result = evaluation.EvaluationResultPart()
    #     for measure, metric in evaluation.metrics_rmse.items():
    #         result.results[measure] = metric.calculate(y_test, y_pred, x_test)
    #
    #     if prnt:
    #         print('MD: ', result)
    #
    #     return result
    #
    # def test(self, x_test, y_test, prnt=False):
    #     result = evaluation.EvaluationResultPart()
    #
    #     if prnt:
    #         print('Results of testing:')
    #     result.model_mf = self.test_cf(x_test, y_test, prnt)
    #     result.model_cs = self.test_md(x_test, y_test, prnt)
    #
    #     return result

    def _concat_data(self, inds_u_x, inds_i_x, y_x, shuffle=True):

        inds_u_xtrain = np.concatenate((inds_u_x, self.x_train[0]))
        inds_i_xtrain = np.concatenate((inds_i_x, self.x_train[1]))
        y_xtrain = np.concatenate((y_x, self.y_train))

        if shuffle:
            order = np.arange(self.n_train + len(y_x))
            np.random.shuffle(order)
            inds_u_xtrain = inds_u_xtrain[order]
            inds_i_xtrain = inds_i_xtrain[order]
            y_xtrain = y_xtrain[order]

        return inds_u_xtrain, inds_i_xtrain, y_xtrain
