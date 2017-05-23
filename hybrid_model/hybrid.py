from typing import NewType, NamedTuple, List, Type
import numpy as np
import copy

# Keras
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

from keras.optimizers import Optimizer

# Local
from hybrid_model.models import SVDpp, AttributeBias
from hybrid_model.callbacks_custom import EarlyStoppingBestVal
from hybrid_model.index_sampler import IndexSampler
from hybrid_model import evaluation
from hybrid_model.transform import Transformation

Matrix = NewType('Matrix', np.ndarray)


class HybridConfig(NamedTuple):
    """
    Config Class for the Hybrid model
    """

    n_factors: int
    reg_bias_mf: float
    reg_latent: float
    reg_bias_cs: float
    reg_att_bias: float

    implicit_thresh_init: float
    implicit_thresh_xtrain: float

    opt_mf_init: Optimizer
    opt_cs_init: Optimizer
    opt_mf_xtrain: Optimizer
    opt_cs_xtrain: Optimizer

    batch_size_init_mf: int
    batch_size_init_cs: int
    batch_size_xtrain_mf: int
    batch_size_xtrain_cs: int
    val_split_init: float
    val_split_xtrain: float

    index_sampler: Type[IndexSampler]

    xtrain_patience: int
    xtrain_max_epochs: int
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

        # Build models
        self.model_mf = SVDpp(self.n_users, self.n_items, config.n_factors, config.reg_latent, config.reg_bias_mf,
                              config.implicit_thresh_init, config.implicit_thresh_xtrain, config.transformation)
        self.model_cs = AttributeBias(meta_users, meta_items, config.reg_att_bias, config.reg_bias_cs,
                                      config.transformation)

        # Callbacks for early stopping during one cross train iteration
        self.callbacks_mf = [EarlyStoppingBestVal('val_loss', patience=4)]
        self.callbacks_cs = [EarlyStoppingBestVal('val_loss', patience=4)]

        # Init to be training data
        self.x_train: List[Matrix] = None
        self.y_train: Matrix = None
        self.n_train: int = 0

        self.user_dist: Matrix = None
        self.item_dist: Matrix = None

        self.index_sampler: IndexSampler = None

    def fit(self, x_train, y_train, x_test=None, y_test=None):

        self.fit_init_only(x_train, y_train, x_test, y_test)
        self.fit_xtrain_only(x_train, y_train, x_test, y_test)

    def fit_init_only(self, x_train, y_train, x_test=None, y_test=None):
        y_train = self.config.transformation.transform(y_train)

        self.x_train = x_train
        self.y_train = y_train
        self.n_train = len(y_train)

        # Get distribution of ratings per user and item
        self.user_dist = np.bincount(x_train[0], minlength=self.n_users)
        self.item_dist = np.bincount(x_train[1], minlength=self.n_items)

        # Init Index Sampler
        self.index_sampler = self.config.index_sampler(self.user_dist, self.item_dist, self.x_train)

        # Set initial global bias value to mean of training data
        mean = np.mean(y_train).astype(np.float32)
        self.model_mf.model.get_layer('bias').weights[0].set_value([mean])
        self.model_cs.model.get_layer('bias').weights[1].set_value([mean])

        # Compile model using optimizer used for initial training
        self.model_mf.compile(self.config.opt_mf_init)
        self.model_cs.compile(self.config.opt_cs_init)

        # Initially train models separately
        self._train_init(x_test, y_test)

    def fit_xtrain_only(self, x_train, y_train, x_test=None, y_test=None):
        # Recompile model using optimizer for cross training
        self.model_mf.compile(self.config.opt_mf_xtrain)
        self.model_cs.compile(self.config.opt_cs_xtrain)

        # Run Cross-Training
        self._train_xtrain(x_test, y_test)

    def _train_init(self, x_test=None, y_test=None):

        # Initial early stopping callbacks
        callbacks_mf = [EarlyStoppingBestVal('val_loss', patience=10)]
        callbacks_cs = [EarlyStoppingBestVal('val_loss', patience=10)]

        # Compute implicit matrix for matrix factorization
        self.model_mf.recompute_implicit(self.x_train, self.y_train, True)

        # Train both models with the training data only
        self.model_mf.model.fit(self.x_train, self.y_train, batch_size=self.config.batch_size_init_mf, epochs=200,
                          validation_split=self.config.val_split_init, verbose=self.verbose, callbacks=callbacks_mf)
        self.model_cs.model.fit(self.x_train, self.y_train, batch_size=self.config.batch_size_init_cs, epochs=200,
                          validation_split=self.config.val_split_init, verbose=self.verbose, callbacks=callbacks_cs)

        # Keras fit method "unflattens" arrays on training. Undo this here
        self.x_train[0] = self.x_train[0].flatten()
        self.x_train[1] = self.x_train[1].flatten()

        # Test of data was handed
        if x_test:
            self.test(x_test, y_test, True)

    def _train_xtrain(self, x_test=None, y_test=None):

        # Config for early stopping
        patience = self.config.xtrain_patience
        min_delta = 0.00005

        best_weights_mf = None
        best_weights_cs = None

        vloss_mf_min = float('inf')
        epoch_min = -1

        # Alternating cross-training
        for i in range(20):
            print('Training step {}'.format(i + 1))

            # CS step (MF -> CS, train coldstart model with augmented data by matrix factorization model)
            self._step_cs()

            # MF step (CS -> MF, train matrix factorization model with augmented data by coldstart model)
            # Cross training is subject to early stopping depending on the MF model validation loss
            vloss_mf = self._step_mf()

            # Performance measure through test data
            if x_test:
                print('Results after training step {}:'.format(i + 1))
                result = self.test(x_test, y_test, True)

            # Check for early stopping
            if vloss_mf < vloss_mf_min - min_delta:
                if self.verbose > 0 or True:
                    print('Min valloss {:.4f} at epoch {}'.format(vloss_mf, i + 1))
                vloss_mf_min = vloss_mf
                epoch_min = i
                # best_weights_mf = copy.deepcopy(self.model_mf.model.get_weights())
                # best_weights_cs = copy.deepcopy(self.model_cs.model.get_weights())
            else:
                if self.verbose > 0 or True:
                    print('Valloss {:.4f} at epoch {}'.format(vloss_mf, i + 1))

                if i >= epoch_min + patience:
                    print('Stopping crosstraining after epoch {}'.format(i + 1))
                    break

            if i + 1 >= self.config.xtrain_max_epochs:
                break

        # Set weights of best epoch
        # self.model_mf.model.set_weights(best_weights_mf)
        # self.model_cs.model.set_weights(best_weights_cs)

    def _step_mf(self):
        # Get indices for cross training from sampling distribution
        inds_u_x, inds_i_x = self.index_sampler.get_indices_from_cs()

        # Get prediction on sampled indices
        y_x = self.model_cs.model.predict([inds_u_x, inds_i_x]).flatten()

        # Recompute implicit matrix
        self.model_mf.recompute_implicit([inds_u_x, inds_i_x], y_x)

        # Combine data with original training data
        inds_u_xtrain, inds_i_xtrain, y_xtrain = self._concat_data(inds_u_x, inds_i_x, y_x, self.config.xtrain_data_shuffle)

        # Update-train MF model with cross-train data
        history = self.model_mf.model.fit([inds_u_xtrain, inds_i_xtrain], y_xtrain, batch_size=self.config.batch_size_xtrain_mf,
                                    epochs=150, validation_split=self.config.val_split_xtrain, verbose=self.verbose,
                                    callbacks=self.callbacks_mf)

        # Return best validation loss
        return min(history.history['val_loss'])

    def _step_cs(self):
        # Get indices for cross training from sampling distribution
        inds_u_x, inds_i_x = self.index_sampler.get_indices_from_mf()

        # Get prediction on sampled indices
        y_x = self.model_mf.model.predict([inds_u_x, inds_i_x]).flatten()

        # Combine data with original training data
        inds_u_xtrain, inds_i_xtrain, y_xtrain = self._concat_data(inds_u_x, inds_i_x, y_x, self.config.xtrain_data_shuffle)

        # Update-train ANN model with cross-train data
        history = self.model_cs.model.fit([inds_u_xtrain, inds_i_xtrain], y_xtrain, batch_size=self.config.batch_size_xtrain_cs,
                                    epochs=150, validation_split=self.config.val_split_xtrain, verbose=self.verbose,
                                    callbacks=self.callbacks_cs)

        # Return best validation loss
        return min(history.history['val_loss'])

    def test_mf(self, x_test, y_test, prnt=False):
        y_pred = self.model_mf.predict(x_test)

        result = evaluation.EvaluationResultPart()
        for measure, metric in evaluation.metrics_rmse.items():
            result.results[measure] = metric.calculate(y_test, y_pred, x_test)

        if prnt:
            print('MF: ', result)

        return result

    def test_cs(self, x_test, y_test, prnt=False):
        y_pred = self.model_cs.predict(x_test)

        result = evaluation.EvaluationResultPart()
        for measure, metric in evaluation.metrics_rmse.items():
            result.results[measure] = metric.calculate(y_test, y_pred, x_test)

        if prnt:
            print('CS: ', result)

        return result

    def test(self, x_test, y_test, prnt=False):
        result = evaluation.EvaluationResultPart()

        if prnt:
            print('Results of testing:')
        result.model_mf = self.test_mf(x_test, y_test, prnt)
        result.model_cs = self.test_cs(x_test, y_test, prnt)

        return result

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
