from keras.layers import Embedding, Input, Flatten
from keras.layers.merge import Concatenate, Dot
from keras.models import Model
from keras.regularizers import l2

from util.layers_custom import BiasLayer
from hybrid_model.models.abstract import AbstractModelCF, bias_init


class SVD(AbstractModelCF):
    def __init__(self, n_users, n_items, config=None):
        super().__init__(n_users, n_items, config)

        # Defaults
        default = {'n_factors': 40, 'reg_bias': 0.00005, 'reg_latent': 0.00003}
        default.update(self.config)
        self.config = default

        n_factors = self.config['n_factors']
        reg_bias = l2(self.config['reg_bias'])
        reg_latent = l2(self.config['reg_latent'])

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
