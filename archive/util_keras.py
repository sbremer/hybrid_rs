import types
import tempfile
import keras.models

from util import BiasLayer


def make_keras_picklable():
    def __getstate__(self):
        json_string = self.to_json()

        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            self.save_weights(fd.name, overwrite=True)
            weights = fd.read()

        return { 'model_str': json_string, 'weights': weights }

    def __setstate__(self, state):
        from keras.models import model_from_json
        model = model_from_json(state['model_str'], custom_objects={'BiasLayer': BiasLayer})
        self.__dict__ = model.__dict__

        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            self.load_weights(fd.name)

    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


def make_keras_picklable_h5():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__