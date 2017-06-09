import numpy as np
from keras.initializers import Constant
from keras.layers import Embedding, Input, Dense, Flatten
from keras.layers.merge import Dot, Concatenate, Add, Multiply
from keras.models import Model
from keras.regularizers import l1, l2

from hybrid_model.transform import TransformationLinear
from util.layers_custom import BiasLayer
from util.callbacks_custom import EarlyStoppingBestVal

bias_init = Constant(0.5)
early_stopping_callback = [EarlyStoppingBestVal('val_loss', patience=10)]


class AbstractModel:
    def __init__(self, n_users, n_items, config, transformation):
        self.model: Model = None
        self.n_users = n_users
        self.n_items = n_items

        if config is None:
            config = {}
        self.config = config
        self.optimizer = self.config.get('optimizer', 'adagrad')
        self.transformation = transformation

    def compile(self, optimizer=None):
        if optimizer:
            self.model.compile(optimizer, 'mse')
        else:
            self.model.compile(self.optimizer, 'mse')

    def fit(self, x_train, y_train, **kwargs):
        y_train = self.transformation.transform(y_train)

        kwargs_default = dict(batch_size=512, epochs=200, validation_split=0.05, verbose=0,
                              callbacks=early_stopping_callback)
        kwargs_default.update(kwargs)

        return self.model.fit(x_train, y_train, **kwargs_default)

    def predict(self, x_test, **kwargs):
        y_pred = self.model.predict(x_test, **kwargs).flatten()

        y_pred = np.maximum(0.0, y_pred)
        y_pred = np.minimum(1.0, y_pred)
        y_pred = self.transformation.invtransform(y_pred)

        return y_pred


class AbstractModelCF(AbstractModel):
    def __init__(self, n_users, n_items, config, transformation):
        super().__init__(n_users, n_items, config, transformation)


class AbstractModelMD(AbstractModel):
    def __init__(self, meta_users, meta_items, config, transformation):
        n_users, self.n_users_features = meta_users.shape[:2]
        n_items, self.n_items_feature = meta_items.shape[:2]
        super().__init__(n_users, n_items, config, transformation)


class BiasEstimator(AbstractModelCF):
    def __init__(self, n_users, n_items, config=None, transformation=TransformationLinear()):
        super().__init__(n_users, n_items, config, transformation)

        reg_bias = l1(self.config['reg_bias'])

        input_u = Input((1,))
        input_i = Input((1,))

        bias_u = Embedding(n_users, 1, input_length=1, embeddings_initializer='zeros',
                           embeddings_regularizer=reg_bias)(input_u)
        bias_u_r = Flatten()(bias_u)
        bias_i = Embedding(n_items, 1, input_length=1, embeddings_initializer='zeros',
                           embeddings_regularizer=reg_bias)(input_i)
        bias_i_r = Flatten()(bias_i)

        added = Concatenate()([bias_u_r, bias_i_r])
        bias_out = BiasLayer(bias_initializer=bias_init)(added)

        self.model = Model(inputs=[input_u, input_i], outputs=bias_out)

        self.compile()


class BiasEstimatorCustom(AbstractModelCF):
    def __init__(self, n_users, n_items, config=None, transformation=TransformationLinear()):
        super().__init__(n_users, n_items, config, transformation)

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


class SVD(AbstractModelCF):
    def __init__(self, n_users, n_items, config=None, transformation=TransformationLinear()):
        super().__init__(n_users, n_items, config, transformation)

        n_factors = self.config.get('n_factors', 35)
        reg_latent = l2(self.config.get('reg_latent', 0.00001))
        reg_bias = l2(self.config.get('reg_bias', 0.00003))

        input_u = Input((1,))
        input_i = Input((1,))

        vec_u = Embedding(self.n_users, n_factors, input_length=1, embeddings_regularizer=reg_latent)(input_u)
        vec_u_r = Flatten()(vec_u)
        vec_i = Embedding(self.n_items, n_factors, input_length=1, embeddings_regularizer=reg_latent)(input_i)
        vec_i_r = Flatten()(vec_i)

        mf = Dot(1)([vec_u_r, vec_i_r])

        bias_u = Embedding(self.n_users, 1, input_length=1, embeddings_initializer='zeros',
                           embeddings_regularizer=reg_bias)(input_u)
        bias_u_r = Flatten()(bias_u)
        bias_i = Embedding(self.n_items, 1, input_length=1, embeddings_initializer='zeros',
                           embeddings_regularizer=reg_bias)(input_i)
        bias_i_r = Flatten()(bias_i)

        added = Concatenate()([bias_u_r, bias_i_r, mf])

        mf_out = BiasLayer(bias_initializer=bias_init, name='bias')(added)

        self.model = Model(inputs=[input_u, input_i], outputs=mf_out)

        self.compile()


class SVDpp(AbstractModelCF):
    def __init__(self, n_users, n_items, config=None, transformation=TransformationLinear()):
        super().__init__(n_users, n_items, config, transformation)

        self.implicit = np.zeros((self.n_users, self.n_items))

        n_factors = self.config.get('n_factors', 35)
        reg_bias = l2(self.config.get('reg_bias', 0.00001))
        reg_latent = l2(self.config.get('reg_latent', 0.00003))

        self.implicit_thresh = self.config.get('implicit_thresh', 4.0)
        self.implicit_thresh_crosstrain = self.config.get('implicit_thresh_crosstrain', 4.75)

        input_u = Input((1,))
        input_i = Input((1,))

        vec_u = Embedding(self.n_users, n_factors, input_length=1, embeddings_regularizer=reg_latent)(input_u)
        vec_u_r = Flatten()(vec_u)
        vec_i = Embedding(self.n_items, n_factors, input_length=1, embeddings_regularizer=reg_latent)(input_i)
        vec_i_r = Flatten()(vec_i)

        vec_implicit = Embedding(self.n_users, self.n_items, input_length=1, trainable=False, name='implicit')(
            input_u)

        implicit_factors = Dense(n_factors, kernel_initializer='normal', activation='linear',
                                 kernel_regularizer=reg_latent)(vec_implicit)

        implicit_factors = Flatten()(implicit_factors)

        vec_u_added = Add()([vec_u_r, implicit_factors])

        mf = Dot(1)([vec_u_added, vec_i_r])

        bias_u = Embedding(self.n_users, 1, input_length=1, embeddings_initializer='zeros',
                           embeddings_regularizer=reg_bias)(input_u)
        bias_u_r = Flatten()(bias_u)
        bias_i = Embedding(self.n_items, 1, input_length=1, embeddings_initializer='zeros',
                           embeddings_regularizer=reg_bias)(input_i)
        bias_i_r = Flatten()(bias_i)

        added = Concatenate()([bias_u_r, bias_i_r, mf])

        mf_out = BiasLayer(bias_initializer=bias_init, name='bias')(added)

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
                self.implicit[u, i] = 1.0

        # Normalize using sqrt (ref. SVD++ paper)
        implicit_norm = self.implicit / np.sqrt(np.maximum(1, np.sum(self.implicit, axis=1)[:, None]))

        self.model.get_layer('implicit').set_weights([implicit_norm])

    def fit(self, x_train, y_train, **kwargs):
        self.recompute_implicit(x_train, y_train)
        return super().fit(x_train, y_train, **kwargs)


class AttributeBias(AbstractModelMD):
    def __init__(self, meta_users, meta_items, config=None, transformation=TransformationLinear()):
        super().__init__(meta_users, meta_items, config, transformation)

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

        factors_i = Dense(self.n_users_features, kernel_initializer='zeros', activation='linear',
                          kernel_regularizer=reg_att_bias, use_bias=False)(vec_features_i)

        mult = Multiply()([factors_i, vec_features_u])

        bias_u = Embedding(self.n_users, 1, input_length=1, embeddings_initializer='zeros',
                           embeddings_regularizer=reg_bias)(input_u)
        bias_u = Flatten()(bias_u)
        bias_i = Embedding(self.n_items, 1, input_length=1, embeddings_initializer='zeros',
                           embeddings_regularizer=reg_bias)(input_i)
        bias_i = Flatten()(bias_i)

        concat = Concatenate()([bias_u, bias_i, mult])

        cs_out = Dense(1, activation='linear', use_bias=True, bias_initializer=bias_init, name='bias')(concat)
        # cs_out = BiasLayer(name='bias')(concat)

        self.model = Model(inputs=[input_u, input_i], outputs=cs_out)

        # Normalize Genre matrix and set static weights
        meta_items = meta_items / np.maximum(1, np.sum(meta_items, axis=1)[:, None])
        # meta_users = meta_users / np.maximum(1, np.sum(meta_users, axis=1)[:, None])
        self.model.get_layer('users_features').set_weights([meta_users])
        self.model.get_layer('items_features').set_weights([meta_items])

        self.compile()


class AttributeBiasExperimental(AbstractModelMD):
    def __init__(self, meta_users, meta_items, config=None, transformation=TransformationLinear()):
        super().__init__(meta_users, meta_items, config, transformation)

        # reg_bias = l1(self.config.get('reg_bias', 0.000003))
        # reg_att_bias = l1(self.config.get('reg_att_bias', 0.000005))

        reg_bias = l2(self.config.get('reg_bias', 0.0001))
        reg_att_bias = l2(self.config.get('reg_att_bias', 0.0002))

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
        feature_bias = Dense(1, activation='linear', use_bias=False, kernel_regularizer=reg_att_bias)(
            feature_concat)

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

        # Features User x Features Item
        uf_if = Dense(self.n_users_features, kernel_initializer='zeros', activation='linear',
                      kernel_regularizer=reg_att_bias, use_bias=False)(vec_features_i)

        mult_uf_if = Multiply()([uf_if, vec_features_u])

        # User/Item Bias
        bias_u = Embedding(self.n_users, 1, input_length=1, embeddings_initializer='zeros',
                           embeddings_regularizer=reg_bias)(input_u)
        bias_u = Flatten()(bias_u)
        bias_i = Embedding(self.n_items, 1, input_length=1, embeddings_initializer='zeros',
                           embeddings_regularizer=reg_bias)(input_i)
        bias_i = Flatten()(bias_i)

        # Combining
        concat = Concatenate()([bias_u, bias_i, mult_uf_i, mult_u_if, mult_uf_if, feature_bias])

        cs_out = Dense(1, activation='linear', use_bias=True, bias_initializer=bias_init, name='bias')(concat)
        # cs_out = BiasLayer(bias_initializer=bias_init, name='bias')(concat)

        self.model = Model(inputs=[input_u, input_i], outputs=cs_out)

        # Normalize Genre matrix and set static weights
        meta_items = meta_items / np.maximum(1, np.sum(meta_items, axis=1)[:, None])
        meta_users = meta_users / np.maximum(1, np.sum(meta_users, axis=1)[:, None])

        self.model.get_layer('users_features').set_weights([meta_users])
        self.model.get_layer('items_features').set_weights([meta_items])

        self.compile()

# class SVDpp(AbstractModel):
#     def __init__(self, n_users, n_items, n_factors=40, reg_latent=0.00005, reg_bias=0.00005, implicit_thresh_init=0.4,
#                  implicit_thresh_xtrain=0.7, transformation=TransformationLinear()):
#
#         self.transformation = transformation
#
#         self.n_users = n_users
#         self.n_items = n_items
#
#         self.implicit = np.zeros((self.n_users, self.n_items))
#
#         self.implicit_thresh_init = implicit_thresh_init
#         self.implicit_thresh_xtrain = implicit_thresh_xtrain
#
#         input_u = Input((1,))
#         input_i = Input((1,))
#
#         vec_u = Embedding(self.n_users, n_factors, input_length=1, embeddings_regularizer=l2(reg_latent))(input_u)
#         vec_u_r = Flatten()(vec_u)
#         vec_i = Embedding(self.n_items, n_factors, input_length=1, embeddings_regularizer=l2(reg_latent))(input_i)
#         vec_i_r = Flatten()(vec_i)
#
#         vec_implicit = Embedding(self.n_users, self.n_items, input_length=1, trainable=False, name='implicit')(input_u)
#
#         implicit_factors = Dense(n_factors, kernel_initializer='normal', activation='linear',
#                                  kernel_regularizer=l2(reg_latent))(vec_implicit)
#
#         implicit_factors = Flatten()(implicit_factors)
#
#         vec_u_added = Add()([vec_u_r, implicit_factors])
#
#         mf = Dot(1)([vec_u_added, vec_i_r])
#
#         bias_u = Embedding(self.n_users, 1, input_length=1, embeddings_initializer='zeros',
#                            embeddings_regularizer=l2(reg_bias))(input_u)
#         bias_u_r = Flatten()(bias_u)
#         bias_i = Embedding(self.n_items, 1, input_length=1, embeddings_initializer='zeros',
#                            embeddings_regularizer=l2(reg_bias))(input_i)
#         bias_i_r = Flatten()(bias_i)
#
#         added = Concatenate()([bias_u_r, bias_i_r, mf])
#
#         mf_out = BiasLayer(name='bias')(added)
#
#         self.model = Model(inputs=[input_u, input_i], outputs=mf_out)
#
#     def recompute_implicit(self, x, y, init=False):
#
#         inds_u, inds_i = x
#
#         if init:
#             thresh = self.implicit_thresh_init
#         else:
#             thresh = self.implicit_thresh_xtrain
#
#         # Use ratings over the threshold as implicit feedback
#         for u, i, r in zip(inds_u, inds_i, y):
#             if r >= thresh:
#                 self.implicit[u, i] = 1.0
#
#         # Normalize using sqrt (ref. SVD++ paper)
#         implicit_norm = self.implicit / np.sqrt(np.maximum(1, np.sum(self.implicit, axis=1)[:, None]))
#
#         self.model.get_layer('implicit').set_weights([implicit_norm])
#
#
# class AttributeBias(AbstractModel):
#     def __init__(self, meta_users, meta_items, reg_att_bias=0.002, reg_bias=0.00003, transformation=TransformationLinear()):
#
#         self.transformation = transformation
#
#         input_u = Input((1,))
#         input_i = Input((1,))
#
#         self.n_users, n_users_features = meta_users.shape[:2]
#         self.n_items, n_items_feature = meta_items.shape[:2]
#
#         vec_features_u = Embedding(self.n_users, n_users_features, input_length=1, trainable=False,
#                                    name='users_features')(input_u)
#         vec_features_u = Flatten()(vec_features_u)
#
#         vec_features_i = Embedding(self.n_items, n_items_feature, input_length=1, trainable=False,
#                                    name='items_features')(input_i)
#         vec_features_i = Flatten()(vec_features_i)
#
#         factors_i = Dense(n_users_features, kernel_initializer='zeros', activation='linear',
#                           kernel_regularizer=l2(reg_att_bias), use_bias=False)(vec_features_i)
#
#         mult = Multiply()([factors_i, vec_features_u])
#
#         bias_u = Embedding(self.n_users, 1, input_length=1, embeddings_initializer='zeros',
#                            embeddings_regularizer=l2(reg_bias))(input_u)
#         bias_u = Flatten()(bias_u)
#         bias_i = Embedding(self.n_items, 1, input_length=1, embeddings_initializer='zeros',
#                            embeddings_regularizer=l2(reg_bias))(input_i)
#         bias_i = Flatten()(bias_i)
#
#         concat = Concatenate()([bias_u, bias_i, mult])
#
#         cs_out = Dense(1, activation='linear', use_bias=True, bias_initializer=bias_init, name='bias')(concat)
#         # cs_out = BiasLayer(name='bias')(concat)
#
#         self.model = Model(inputs=[input_u, input_i], outputs=cs_out)
#
#         # Normalize Genre matrix and set static weights
#         meta_items = meta_items / np.maximum(1, np.sum(meta_items, axis=1)[:, None])
#         self.model.get_layer('users_features').set_weights([meta_users])
#         self.model.get_layer('items_features').set_weights([meta_items])
