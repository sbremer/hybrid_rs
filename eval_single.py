import pickle
import numpy as np

np.random.seed(0)

# Local imports
from hybrid_model.hybrid import HybridModel
from hybrid_model.evaluation import Evaluation
import util
from hybrid_model.config import hybrid_config

# (inds_u, inds_i, y, users_features, items_features) = pickle.load(open('data/ml100k.pickle', 'rb'))
(inds_u, inds_i, y, users_features, items_features) = pickle.load(open('data/ml1m.pickle', 'rb'))

n_users, n_users_features = users_features.shape
n_items, n_items_features = items_features.shape

# Crossvalidation
n_fold = 5
user_coldstart = False
if user_coldstart:
    kfold = util.kfold_entries(n_fold, inds_u)
    # kfold = util.kfold_entries_plus(n_fold, inds_u, 2)
else:
    kfold = util.kfold(n_fold, inds_u)

kfold = list(kfold)

xval_train, xval_test = kfold[1]

# Dataset training
inds_u_train = inds_u[xval_train]
inds_i_train = inds_i[xval_train]
y_train = y[xval_train]
n_train = len(y_train)

# Dataset testing
inds_u_test = inds_u[xval_test]
inds_i_test = inds_i[xval_test]
y_test = y[xval_test]


def analyze_hybrid(hybrid_model):
    hybrid_model.fit_init_only([inds_u_train, inds_i_train], y_train)

    print('Hybrid before xtrain:')
    result = evaluation.evaluate_hybrid(hybrid_model, [inds_u_test, inds_i_test], y_test)
    print(result)

    hybrid_model.fit_xtrain_only([inds_u_train, inds_i_train], y_train)

    print('Hybrid after xtrain:')
    result = evaluation.evaluate_hybrid(hybrid_model, [inds_u_test, inds_i_test], y_test)
    print(result)


def analyze_model(model, name):
    model.fit([inds_u_train, inds_i_train], y_train)
    result = evaluation.evaluate(model, [inds_u_test, inds_i_test], y_test)
    print(name)
    print(result)

evaluation = Evaluation()

models = []

# from hybrid_model.baselines import BaselineBias
# model = BaselineBias(n_users, n_items)
# models.append((model.__class__.__name__, model))

# model = HybridModel(users_features, items_features, hybrid_config, verbose=2)
# models.append(('Hybrid', model))

from hybrid_model.baselines import BaselineSVDpp
model = BaselineSVDpp(n_users, n_items, n_factors=20, reg_latent=0.00002)
models.append((model.__class__.__name__, model))

# from hybrid_model.baselines import AttributeBias
# model = AttributeBias(users_features, items_features)
# models.append((model.__class__.__name__, model))

# from hybrid_model.baselines import AttributeBiasExperimental
# model = AttributeBiasExperimental(users_features, items_features)
# models.append((model.__class__.__name__, model))

for name, model in models:
    if type(model) is HybridModel:
        analyze_hybrid(model)
    else:
        analyze_model(model, name)
