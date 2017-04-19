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


def kfold_entries_plus(k, entries, plus):
    unique = np.unique(entries)
    kf = KFold(n_splits=k, shuffle=True)
    for train_indices_entries, test_indices_entries in kf.split(unique):

        train = {key: [] for key in train_indices_entries}
        test = {key: [] for key in test_indices_entries}
        for i, x in enumerate(entries):
            if x in train_indices_entries:
                train[x].append(i)
            if x in test_indices_entries:
                test[x].append(i)

        plus_list = []

        for test_key in test.keys():
            for _ in range(plus):
                n = len(test[test_key])
                if n == 0:
                    break
                i = np.random.randint(n - 1)
                element = test[test_key].pop(i)
                plus_list.append(element)

        train_indices = list(itertools.chain(plus_list, *train.values()))
        test_indices = list(itertools.chain(*test.values()))

        yield train_indices, test_indices


class IndexGen:
    """
    Generates indices for user-item pairs which are not in the training set.
    Build for sparse training set! Might be very inefficient else.
    """

    def __init__(self, n_users, n_items, inds_u, inds_i):
        assert len(inds_u) == len(inds_i)

        self.n_users = n_users
        self.n_items = n_items

        # Create probability distributions for getting data from MF and ANN
        counts = np.bincount(inds_u.astype(np.int32), minlength=n_users)

        counts_mf = np.maximum(counts - 15, 1)
        counts_ann = np.maximum(counts, 4) ** 2

        self.prob_from_mf = counts_mf / sum(counts_mf)
        self.prob_from_ann = (1 / counts_ann) / sum(1 / counts_ann)

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
            u = np.random.choice(np.arange(self.n_users), p=self.prob_from_mf)
            i = np.random.randint(self.n_items)

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
            u = np.random.choice(np.arange(self.n_users), p=self.prob_from_ann)
            i = np.random.randint(self.n_items)

            if (u, i) not in self.lookup and (u, i) not in lookup_samples:
                inds_u[got] = u
                inds_i[got] = i
                lookup_samples[(u, i)] = True
                got += 1

        return inds_u, inds_i


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

from keras.layers import Layer
import keras.backend as K
import itertools
import math


class InputCombinations(Layer):
    def __init__(self, k, **kwargs):
        super(InputCombinations, self).__init__(**kwargs)
        self.k = k

    def build(self, input_shape):
        assert len(input_shape) == 2
        self.built = True

    def call(self, inputs):
        n = inputs._keras_shape[1]
        order = itertools.combinations(range(n), self.k)

        xs = [inputs[:, o] for o in order]
        output = K.stack(xs, axis=2)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        n = input_shape[1]
        k = self.k
        n_combinations = math.factorial(n) // (math.factorial(k) * math.factorial(n-k))

        return (None, n_combinations, k)

    def get_config(self):
        config = {
            'k': self.k,
        }
        base_config = super(InputCombinations, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


from keras.legacy import interfaces
from keras.engine import InputSpec
from keras import activations, initializers, regularizers, constraints


class BiasLayer(Layer):
    """Just your regular densely-connected NN layer.

    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.

    # Example

    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)

        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Dense(32))
    ```

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    @interfaces.legacy_dense_support
    def __init__(self,
                 activation=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 activity_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(BiasLayer, self).__init__(**kwargs)
        self.activation = activations.get(activation)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.bias = self.add_weight((1,),
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        output = K.sum(inputs, axis=-1, keepdims=True)
        output = K.bias_add(output, self.bias)
        return output

        # for i in range(len(inputs)):
        #     output += inputs[i]
        # return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = 1
        return tuple(output_shape)

    def get_config(self):
        config = {
            'activation': activations.serialize(self.activation),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(BiasLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
