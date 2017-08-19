import numpy as np
from keras.layers import Embedding, Input, Flatten, Dense
from keras.layers.merge import Concatenate, Multiply
from keras.models import Model
from keras.regularizers import l2

from hybrid_model.models.abstract import AbstractModelMD, bias_init


class AttributeBiasLight(AbstractModelMD):
    def __init__(self, meta_users, meta_items, config=None):
        super().__init__(meta_users, meta_items, config)

        # Defaults
        default = {'reg_bias': 0.0005, 'reg_att_bias': 0.003}
        default.update(self.config)
        self.config = default

        reg_bias = l2(self.config['reg_bias'])
        reg_att_bias = l2(self.config['reg_att_bias'])

        input_u = Input((1,))
        input_i = Input((1,))

        vec_features_u = Embedding(self.n_users, self.n_users_features, input_length=1, trainable=False,
                                   name='users_features')(input_u)
        vec_features_u = Flatten()(vec_features_u)

        vec_features_i = Embedding(self.n_items, self.n_items_feature, input_length=1, trainable=False,
                                   name='items_features')(input_i)
        vec_features_i = Flatten()(vec_features_i)

        # Features User x Item
        uf_i = Embedding(self.n_items, self.n_users_features, input_length=1, embeddings_regularizer=reg_att_bias,
                         embeddings_initializer='zeros')(input_i)
        uf_i = Flatten()(uf_i)
        mult_uf_i = Multiply()([uf_i, vec_features_u])

        # User x Item Features
        u_if = Embedding(self.n_users, self.n_items_feature, input_length=1, embeddings_regularizer=reg_att_bias,
                         embeddings_initializer='zeros')(input_u)
        u_if = Flatten()(u_if)
        mult_u_if = Multiply()([u_if, vec_features_i])

        # User/Item Bias
        bias_u = Embedding(self.n_users, 1, input_length=1, embeddings_initializer='zeros',
                           embeddings_regularizer=reg_bias)(input_u)
        bias_u = Flatten()(bias_u)
        bias_i = Embedding(self.n_items, 1, input_length=1, embeddings_initializer='zeros',
                           embeddings_regularizer=reg_bias)(input_i)
        bias_i = Flatten()(bias_i)

        # Combining
        concat = Concatenate()([bias_u, bias_i, mult_uf_i, mult_u_if])

        cs_out = Dense(1, activation='linear', use_bias=True, kernel_initializer='ones',
                       bias_initializer=bias_init, name='bias')(concat)
        # cs_out = BiasLayer(bias_initializer=bias_init, name='bias')(concat)

        self.model = Model(inputs=[input_u, input_i], outputs=cs_out)

        # Normalize Genre matrix and set static weights
        meta_items = meta_items / np.maximum(1, np.sum(meta_items, axis=1)[:, None])
        meta_users = meta_users / np.maximum(1, np.sum(meta_users, axis=1)[:, None])

        self.model.get_layer('users_features').set_weights([meta_users])
        self.model.get_layer('items_features').set_weights([meta_items])

        self.compile()
