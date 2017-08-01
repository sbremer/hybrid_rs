import numpy as np
from keras.layers import Embedding, Input, Flatten, Dense
from keras.layers.merge import Concatenate, Dot, Multiply, Add
from keras.models import Model
from keras.regularizers import l2

from hybrid_model.models.abstract import AbstractModelMD, bias_init
from util.layers_custom import BiasLayer


class ExperimentalFactorization(AbstractModelMD):
    def __init__(self, meta_users, meta_items, config=None):
        super().__init__(meta_users, meta_items, config)

        self.implicit = np.zeros((self.n_items, self.n_users,))

        # Defaults
        default = {'n_factors': 40, 'reg_bias': 0.00005, 'reg_latent': 0.00003, 'implicit_thresh': 4.0,
                   'implicit_thresh_crosstrain': 4.75}

        default.update(self.config)
        self.config = default

        n_factors = self.config['n_factors']
        reg_bias = l2(self.config['reg_bias'])
        reg_latent = l2(self.config['reg_latent'])

        self.implicit_thresh = self.config.get('implicit_thresh', 4.0)
        self.implicit_thresh_crosstrain = self.config.get('implicit_thresh_crosstrain', 4.75)

        input_u = Input((1,))
        input_i = Input((1,))

        vec_features_u = Embedding(self.n_users, self.n_users_features, input_length=1, trainable=False,
                                   name='users_features')(input_u)
        vec_features_u = Flatten()(vec_features_u)

        vec_features_i = Embedding(self.n_items, self.n_items_feature, input_length=1, trainable=False,
                                   name='items_features')(input_i)
        vec_features_i = Flatten()(vec_features_i)

        # Feature bias
        feature_concat = Concatenate()([vec_features_u, vec_features_i])
        feature_bias = Dense(1, activation='linear', use_bias=False, kernel_initializer='zeros', kernel_regularizer=reg_bias)(
            feature_concat)

        # Feature Factorization
        lat_features_u = Dense(n_factors, use_bias=False, kernel_regularizer=reg_latent)(vec_features_u)
        lat_features_i = Dense(n_factors, use_bias=False, kernel_regularizer=reg_latent)(vec_features_i)

        lat_features = Multiply()([lat_features_u, lat_features_i])

        vec_u = Embedding(self.n_users, n_factors, input_length=1, embeddings_regularizer=reg_latent)(input_u)
        vec_u_r = Flatten()(vec_u)

        vec_implicit = Embedding(self.n_items, self.n_users, input_length=1, trainable=False, name='implicit')(
            input_i)

        implicit_factors = Dense(n_factors, kernel_initializer='normal', activation='linear',
                                 kernel_regularizer=reg_latent)(vec_implicit)

        implicit_factors = Flatten()(implicit_factors)

        mf_u = Add()([vec_u_r, lat_features_u])
        mf_i = Add()([implicit_factors, lat_features_i])

        mf = Dot(1)([mf_u, mf_i])

        bias_u = Embedding(self.n_users, 1, input_length=1, embeddings_initializer='zeros',
                           embeddings_regularizer=reg_bias)(input_u)
        bias_u_r = Flatten()(bias_u)
        bias_i = Embedding(self.n_items, 1, input_length=1, embeddings_initializer='zeros',
                           embeddings_regularizer=reg_bias)(input_i)
        bias_i_r = Flatten()(bias_i)

        added = Concatenate()([bias_u_r, bias_i_r, mf, feature_bias])

        mf_out = BiasLayer(bias_initializer=bias_init, name='bias', activation='sigmoid')(added)

        self.model = Model(inputs=[input_u, input_i], outputs=mf_out)

        self.compile()

    def recompute_implicit(self, x, y, transformed=False, crosstrain=False):

        if transformed:
            if crosstrain:
                thresh = self.transformation.transform(self.implicit_thresh_crosstrain)
            else:
                thresh = self.transformation.transform(self.implicit_thresh)
        else:
            if crosstrain:
                thresh = self.implicit_thresh_crosstrain
            else:
                thresh = self.implicit_thresh

        inds_u, inds_i = x

        # Use ratings over the threshold as implicit feedback
        for u, i, r in zip(inds_u, inds_i, y):
            if r >= thresh:
                self.implicit[i, u] = 1.0

        # Normalize using sqrt (ref. SVD++ paper)
        implicit_norm = self.implicit / np.sqrt(np.maximum(1, np.sum(self.implicit, axis=1)[:, None]))

        self.model.get_layer('implicit').set_weights([implicit_norm])

    def fit(self, x_train, y_train, **kwargs):
        self.recompute_implicit(x_train, y_train)
        return super().fit(x_train, y_train, **kwargs)
