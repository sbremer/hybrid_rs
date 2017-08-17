import numpy as np
from keras.layers import Embedding, Input, Flatten
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.regularizers import l2

from util.layers_custom import BiasLayer
from hybrid_model.models.abstract import AbstractModelCF, bias_init


class BiasEstimator(AbstractModelCF):
    def __init__(self, n_users, n_items, config=None):
        super().__init__(n_users, n_items, config)

        # Defaults
        default = {'reg_bias': 0.00005, 'include_user': True, 'include_item': True}
        default.update(self.config)
        self.config = default

        reg_bias = l2(self.config['reg_bias'])

        input_u = Input((1,))
        input_i = Input((1,))

        bias_u = Embedding(n_users, 1, input_length=1, embeddings_initializer='zeros',
                           trainable=self.config['include_user'], embeddings_regularizer=reg_bias)(input_u)
        bias_u_r = Flatten()(bias_u)
        bias_i = Embedding(n_items, 1, input_length=1, embeddings_initializer='zeros',
                           trainable=self.config['include_item'], embeddings_regularizer=reg_bias)(input_i)
        bias_i_r = Flatten()(bias_i)

        added = Concatenate()([bias_u_r, bias_i_r])
        bias_out = BiasLayer(bias_initializer=bias_init)(added)

        self.model = Model(inputs=[input_u, input_i], outputs=bias_out)

        self.compile()


class BiasEstimatorCustom(AbstractModelCF):
    def __init__(self, n_users, n_items, config=None):
        super().__init__(n_users, n_items, config)

        self.include_user = self.config.get('include_user', True)
        self.include_item = self.config.get('include_item', True)

    def fit(self, x, y, **kwargs):
        self.global_avg = 0.0
        self.bias_user = np.zeros((self.n_users,))
        self.bias_item = np.zeros((self.n_items,))

        user_r = np.zeros((self.n_users,))
        item_r = np.zeros((self.n_items,))

        inds_u = x[0]
        inds_i = x[1]

        for u, i, r in zip(inds_u, inds_i, y):
            self.global_avg += r
            self.bias_user[u] += r
            user_r[u] += 1.0
            self.bias_item[i] += r
            item_r[i] += 1.0

        self.global_avg /= len(y)

        if self.include_user:
            self.bias_user = self.bias_user / user_r - self.global_avg
            self.bias_user[~np.isfinite(self.bias_user)] = 0.0
        else:
            self.bias_user = np.zeros((self.n_users,))

        if self.include_item:
            self.bias_item = self.bias_item / item_r - self.global_avg
            self.bias_item[~np.isfinite(self.bias_item)] = 0.0
        else:
            self.bias_item = np.zeros((self.n_items,))

    def predict(self, x_test, **kwargs):
        inds_u = x_test[0]
        inds_i = x_test[1]

        y = np.zeros((len(inds_u, )))

        for ind, (u, i) in enumerate(zip(inds_u, inds_i)):
            y[ind] = self.global_avg + self.bias_user[u] + self.bias_item[i]

        return y