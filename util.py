import numpy as np
from sklearn.model_selection import KFold
from keras.callbacks import Callback
import warnings
import copy


def kfold(k, entries):
    kf = KFold(n_splits=k, shuffle=True)
    return kf.split(entries)


def kfold_entries(k, entries):
    unique = np.unique(entries)
    kf = KFold(n_splits=k, shuffle=True)
    for train_indices_entries, test_indices_entries in kf.split(unique):
        train_indices = [i for i, x in enumerate(entries) if x in train_indices_entries]
        test_indices = [i for i, x in enumerate(entries) if x in test_indices_entries]
        yield train_indices, test_indices


class IndexGen:
    """
    Generates indices for user-item pairs which are not in the training set.
    Build for sparse training set! Might be very inefficient else.
    """

    def __init__(self, n_users, n_items, U, I):
        assert len(U) == len(I)

        self.n_users = n_users
        self.n_items = n_items

        # Create and fill entry lookup table
        self.lookup = {}
        for u_i in zip(U, I):
            self.lookup[u_i] = True

    def get_indices(self, n_indices):
        U = np.zeros((n_indices,))
        I = np.zeros((n_indices,))

        got = 0
        while got < n_indices:
            u = np.random.randint(self.n_users)
            i = np.random.randint(self.n_items)
            if (u, i) not in self.lookup:
                U[got] = u
                I[got] = i
                got += 1

        return U, I


class EarlyStoppingBestVal(Callback):
    """Stop training when a monitored quantity has stopped improving.
    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
    """

    def __init__(self, monitor='val_loss',
                 min_delta=0, patience=0, verbose=0, mode='auto'):
        super(EarlyStoppingBestVal, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        self.wait = 0  # Allow instances to be re-used
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_model_epoch = -1
        self.best_model_weights = copy.deepcopy(self.model.get_weights())

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Early stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            self.best_model_epoch = epoch
            self.best_model_weights = copy.deepcopy(self.model.get_weights())
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
            self.wait += 1

    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_model_weights)

        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping. Using weights from epoch %d according to lowest validation error' % (self.stopped_epoch, self.best_model_epoch))
