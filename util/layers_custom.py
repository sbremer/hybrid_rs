from keras.layers import Layer
import keras.backend as K
from keras.legacy import interfaces
from keras.engine import InputSpec
from keras import activations, initializers, regularizers, constraints


class BiasLayer(Layer):
    """
    Adaped from Dense layer, just not using input weights, only summing up inputs and adding a trainable bias.
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


class MultiplyVector(Layer):
    @interfaces.legacy_dense_support
    def __init__(self,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(MultiplyVector, self).__init__(**kwargs)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight((input_dim,),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        output = inputs * self.kernel
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        assert input_shape[-1]
        return input_shape

    def get_config(self):
        config = {
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        base_config = super(MultiplyVector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
