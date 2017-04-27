import numpy as np

# Keras
from keras.layers import Embedding, Input, Dense, Flatten
from keras.layers.merge import Dot, Concatenate, Add, Multiply
from keras.regularizers import l2
from keras.models import Model

# Local
from util import BiasLayer


class AbstractKerasModel:
    def __init__(self):
        self.model: Model = None

    def compile(self, optimizer):
        self.model.compile(optimizer, 'mse')

    def fit(self, x_train, y_train, **kwargs):
        return self.model.fit(x_train, y_train, **kwargs)

    def predict(self, x_test, **kwargs):
        return self.model.predict(x_test, **kwargs).flatten()


class SVDpp(AbstractKerasModel):
    def __init__(self, n_users, n_items, n_factors=40, reg_latent=0.00005, reg_bias=0.00005, implicit_thresh_init=0.4, implicit_thresh_xtrain=0.7):

        self.n_users = n_users
        self.n_items = n_items

        self.implicit = np.zeros((self.n_users, self.n_items))

        self.implicit_thresh_init = implicit_thresh_init
        self.implicit_thresh_xtrain = implicit_thresh_xtrain

        input_u = Input((1,))
        input_i = Input((1,))

        vec_u = Embedding(self.n_users, n_factors, input_length=1, embeddings_regularizer=l2(reg_latent))(input_u)
        vec_u_r = Flatten()(vec_u)
        vec_i = Embedding(self.n_items, n_factors, input_length=1, embeddings_regularizer=l2(reg_latent))(input_i)
        vec_i_r = Flatten()(vec_i)

        vec_implicit = Embedding(self.n_users, self.n_items, input_length=1, trainable=False, name='implicit')(input_u)

        implicit_factors = Dense(n_factors, kernel_initializer='normal', activation='linear',
                                 kernel_regularizer=l2(reg_latent))(vec_implicit)

        implicit_factors = Flatten()(implicit_factors)

        vec_u_added = Add()([vec_u_r, implicit_factors])

        mf = Dot(1)([vec_u_added, vec_i_r])

        bias_u = Embedding(self.n_users, 1, input_length=1, embeddings_regularizer=l2(reg_bias))(input_u)
        bias_u_r = Flatten()(bias_u)
        bias_i = Embedding(self.n_items, 1, input_length=1, embeddings_regularizer=l2(reg_bias))(input_i)
        bias_i_r = Flatten()(bias_i)

        added = Concatenate()([bias_u_r, bias_i_r, mf])

        mf_out = BiasLayer()(added)

        self.model = Model(inputs=[input_u, input_i], outputs=mf_out)

    def recompute_implicit(self, x, y, init=False):

        inds_u, inds_i = x

        if init:
            thresh = self.implicit_thresh_init
        else:
            thresh = self.implicit_thresh_xtrain

        # Use ratings over the threshold as implicit feedback
        for u, i, r in zip(inds_u, inds_i, y):
            if r >= thresh:
                self.implicit[u, i] = 1.0

        # Normalize using sqrt (ref. SVD++ paper)
        implicit_norm = self.implicit / np.sqrt(np.maximum(1, np.sum(self.implicit, axis=1)[:, None]))

        self.model.get_layer('implicit').set_weights([implicit_norm])


class AttributeBias(AbstractKerasModel):
    def __init__(self, meta_users, meta_items, reg_att_bias=0.002, reg_bias=0.00005):
        input_u = Input((1,))
        input_i = Input((1,))

        self.n_users, n_users_features = meta_users.shape[:2]
        self.n_items, n_items_feature = meta_items.shape[:2]

        vec_features_u = Embedding(self.n_users, n_users_features, input_length=1, trainable=False,
                                   name='users_features')(input_u)
        vec_features_u = Flatten()(vec_features_u)

        vec_features_i = Embedding(self.n_items, n_items_feature, input_length=1, trainable=False,
                                   name='items_features')(input_i)
        vec_features_i = Flatten()(vec_features_i)

        factors_i = Dense(n_users_features, kernel_initializer='normal', activation='linear',
                          kernel_regularizer=l2(reg_att_bias), use_bias=False)(vec_features_i)

        mult = Multiply()([factors_i, vec_features_u])

        bias_u = Embedding(self.n_users, 1, input_length=1, embeddings_initializer='zeros',
                           embeddings_regularizer=l2(reg_bias))(input_u)
        bias_u = Flatten()(bias_u)
        bias_i = Embedding(self.n_items, 1, input_length=1, embeddings_initializer='zeros',
                           embeddings_regularizer=l2(reg_bias))(input_i)
        bias_i = Flatten()(bias_i)

        concat = Concatenate()([bias_u, bias_i, mult])

        cs_out = Dense(1)(concat)

        self.model = Model(inputs=[input_u, input_i], outputs=cs_out)

        # Normalize Genre matrix and set static weights
        meta_items = meta_items / np.maximum(1, np.sum(meta_items, axis=1)[:, None])
        self.model.get_layer('users_features').set_weights([meta_users])
        self.model.get_layer('items_features').set_weights([meta_items])

