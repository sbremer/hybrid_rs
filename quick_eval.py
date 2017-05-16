import pickle
import sys

from hybrid_model.util import make_keras_picklable
make_keras_picklable()

sys.setrecursionlimit(10000)
# pickle.dump((model, [inds_u_test, inds_u_test], y_test), open('data/quick_eval.pickle', 'wb'))


# model = pickle.load(open('data/model.pickle', 'rb'))
model, x_test, y_test = pickle.load(open('data/quick_eval.pickle', 'rb'))

result = model.evaluate(x_test, y_test)
print(result)
