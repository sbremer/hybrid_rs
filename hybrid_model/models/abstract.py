import numpy as np
from keras.models import Model
from keras.initializers import Constant

from util.callbacks_custom import EarlyStoppingBestVal
from hybrid_model.transform import TransformationLinear

early_stopping_callback = [EarlyStoppingBestVal('val_loss', patience=10)]
bias_init = Constant(0.5)


class AbstractModel:
    def __init__(self, n_users, n_items, config):
        self.model: Model = None
        self.n_users = n_users
        self.n_items = n_items

        if config is None:
            config = {}
        self.config = config

        self.optimizer = self.config.get('optimizer', 'adagrad')
        self.transformation = self.config.get('transformation', TransformationLinear())

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
    def __init__(self, n_users, n_items, config):
        super().__init__(n_users, n_items, config)


class AbstractModelMD(AbstractModel):
    def __init__(self, meta_users, meta_items, config):
        n_users, self.n_users_features = meta_users.shape[:2]
        n_items, self.n_items_feature = meta_items.shape[:2]
        super().__init__(n_users, n_items, config)