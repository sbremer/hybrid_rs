from typing import NewType, NamedTuple, List, Type, Dict
import numpy as np
import copy

# Keras
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

from keras.optimizers import Optimizer

# Local
from hybrid_model.models import AbstractModelCF, AbstractModelMD
from hybrid_model.callbacks_custom import EarlyStoppingBestVal
from hybrid_model.index_sampler import IndexSampler
from hybrid_model import evaluation
from hybrid_model.transform import Transformation

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

        type_cf = config.model_type_cf
        config_cf = config.model_config_cf
        type_md = config.model_type_md
        config_md = config.model_config_md
        tranfo = config.transformation

        # Build models
        self.model_cf = type_cf(self.n_users, self.n_items, config_cf, tranfo)
        self.model_md = type_md(meta_users, meta_items, config_md, tranfo)

        # Callbacks for early stopping during one cross train iteration
        self.callbacks_cf = [EarlyStoppingBestVal('val_loss', patience=4)]
        self.callbacks_md = [EarlyStoppingBestVal('val_loss', patience=4)]

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

        # # Set initial global bias value to mean of training data
        # mean = np.mean(y_train).astype(np.float32)
        # self.model_cf.model.get_layer('bias').weights[0].set_value([mean])
        # self.model_md.model.get_layer('bias').weights[1].set_value([mean])

        # Compile model using optimizer used for initial training
        self.model_cf.compile(self.config.opt_cf_init)
        self.model_md.compile(self.config.opt_md_init)

        # Initially train models separately
        self._train_init(x_test, y_test)

    def fit_xtrain_only(self, x_test=None, y_test=None):
        # Recompile model using optimizer for cross training
        self.model_cf.compile(self.config.opt_cf_xtrain)
        self.model_md.compile(self.config.opt_md_xtrain)

        # Run Cross-Training
        self._train_xtrain(x_test, y_test)

    def _train_init(self, x_test=None, y_test=None):

        # Initial early stopping callbacks
        callbacks_cf = [EarlyStoppingBestVal('val_loss', patience=10)]
        callbacks_md = [EarlyStoppingBestVal('val_loss', patience=10)]

        # Compute implicit matrix for matrix factorization
        self.model_cf.recompute_implicit(self.x_train, self.y_train, thresh=0.7)

        # Train both models with the training data only
        self.model_cf.model.fit(self.x_train, self.y_train, batch_size=self.config.batch_size_init_cf, epochs=200,
                                validation_split=self.config.val_split_init, verbose=self.verbose, callbacks=callbacks_cf)
        self.model_md.model.fit(self.x_train, self.y_train, batch_size=self.config.batch_size_init_md, epochs=200,
                                validation_split=self.config.val_split_init, verbose=self.verbose, callbacks=callbacks_md)

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

        vloss_cf_min = float('inf')
        epoch_min = -1

        # Alternating cross-training
        for i in range(20):
            print('Training step {}'.format(i + 1))

            self._step_cf_md()

            # MF step (CS -> MF, train matrix factorization model with augmented data by coldstart model)
            # Cross training is subject to early stopping depending on the MF model validation loss
            vloss_cf = self._step_md_cf()

            # Performance measure through test data
            if x_test:
                print('Results after training step {}:'.format(i + 1))
                result = self.test(x_test, y_test, True)

            # Check for early stopping
            if vloss_cf < vloss_cf_min - min_delta:
                if self.verbose > 0 or True:
                    print('Min valloss {:.4f} at epoch {}'.format(vloss_cf, i + 1))
                vloss_cf_min = vloss_cf
                epoch_min = i
                # best_weights_mf = copy.deepcopy(self.model_cf.model.get_weights())
                # best_weights_cs = copy.deepcopy(self.model_md.model.get_weights())
            else:
                if self.verbose > 0 or True:
                    print('Valloss {:.4f} at epoch {}'.format(vloss_cf, i + 1))

                if i >= epoch_min + patience:
                    print('Stopping crosstraining after epoch {}'.format(i + 1))
                    break

            if i + 1 >= self.config.xtrain_max_epochs:
                break

        # Set weights of best epoch
        # self.model_cf.model.set_weights(best_weights_mf)
        # self.model_md.model.set_weights(best_weights_cs)

    def _step_md_cf(self):
        # Get indices for cross training from sampling distribution
        inds_u_x, inds_i_x = self.index_sampler.get_indices_from_md()

        # Get prediction on sampled indices
        y_x = self.model_md.model.predict([inds_u_x, inds_i_x]).flatten()

        # Recompute implicit matrix
        self.model_cf.recompute_implicit([inds_u_x, inds_i_x], y_x, thresh=0.7)

        # Combine data with original training data
        inds_u_xtrain, inds_i_xtrain, y_xtrain = self._concat_data(inds_u_x, inds_i_x, y_x, self.config.xtrain_data_shuffle)

        # Update-train MF model with cross-train data
        history = self.model_cf.model.fit([inds_u_xtrain, inds_i_xtrain], y_xtrain, batch_size=self.config.batch_size_xtrain_cf,
                                          epochs=150, validation_split=self.config.val_split_xtrain, verbose=self.verbose,
                                          callbacks=self.callbacks_cf)

        # Return best validation loss
        return min(history.history['val_loss'])

    def _step_cf_md(self):
        # Get indices for cross training from sampling distribution
        inds_u_x, inds_i_x = self.index_sampler.get_indices_from_cf()

        # Get prediction on sampled indices
        y_x = self.model_cf.model.predict([inds_u_x, inds_i_x]).flatten()

        # Combine data with original training data
        inds_u_xtrain, inds_i_xtrain, y_xtrain = self._concat_data(inds_u_x, inds_i_x, y_x, self.config.xtrain_data_shuffle)

        # Update-train ANN model with cross-train data
        history = self.model_md.model.fit([inds_u_xtrain, inds_i_xtrain], y_xtrain, batch_size=self.config.batch_size_xtrain_md,
                                          epochs=150, validation_split=self.config.val_split_xtrain, verbose=self.verbose,
                                          callbacks=self.callbacks_md)

        # Return best validation loss
        return min(history.history['val_loss'])

    def test_cf(self, x_test, y_test, prnt=False):
        y_pred = self.model_cf.predict(x_test)

        result = evaluation.EvaluationResultPart()
        for measure, metric in evaluation.metrics_rmse.items():
            result.results[measure] = metric.calculate(y_test, y_pred, x_test)

        if prnt:
            print('CF: ', result)

        return result

    def test_md(self, x_test, y_test, prnt=False):
        y_pred = self.model_md.predict(x_test)

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
        result.model_mf = self.test_cf(x_test, y_test, prnt)
        result.model_cs = self.test_md(x_test, y_test, prnt)

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
