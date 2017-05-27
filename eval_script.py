import pickle
import numpy as np
from typing import NamedTuple, List, Dict, Tuple, Type

np.random.seed(0)

# Local imports
from hybrid_model.hybrid import HybridModel
from hybrid_model.evaluation import Evaluation, EvaluationResult, EvaluationResultHybrid
from hybrid_model.models import AbstractModel, AbstractModelCF, AbstractModelMD
from hybrid_model.dataset import Dataset
import util


class EvalModel(NamedTuple):
    name: str
    model_type: Type[AbstractModel]
    config: Dict


def _analyze_hybrid(hybrid_model, evaluation: Evaluation, train, test)\
        -> Tuple[EvaluationResultHybrid, EvaluationResultHybrid]:

    hybrid_model.fit_init_only(*train)
    result_before_x = evaluation.evaluate_hybrid(hybrid_model, *test)

    hybrid_model.fit_xtrain_only(*train)
    result_after_x = evaluation.evaluate_hybrid(hybrid_model, *test)

    return result_before_x, result_after_x


def _analyze_model(model, evaluation: Evaluation, train, test)\
        -> EvaluationResult:
    model.fit(*train)
    result = evaluation.evaluate(model, *test)
    return result


def evaluate_models_single(dataset: Dataset, models: List[EvalModel], user_coldstart=False):
    (inds_u, inds_i, y, users_features, items_features) = dataset.data

    # Crossvalidation
    n_fold = 5
    if user_coldstart:
        kfold = util.kfold_entries(n_fold, inds_u)
        # kfold = util.kfold_entries_plus(n_fold, inds_u, 2)
    else:
        kfold = util.kfold(n_fold, inds_u)

    kfold = list(kfold)

    xval_train, xval_test = kfold[0]

    # Dataset training
    inds_u_train = inds_u[xval_train]
    inds_i_train = inds_i[xval_train]
    y_train = y[xval_train]
    n_train = len(y_train)

    # Dataset testing
    inds_u_test = inds_u[xval_test]
    inds_i_test = inds_i[xval_test]
    y_test = y[xval_test]

    evaluation = Evaluation()

    train = ([inds_u_train, inds_i_train], y_train)
    test = ([inds_u_test, inds_i_test], y_test)

    results_single = []

    for name, model_type, config in models:

        if issubclass(model_type, HybridModel):
            model = model_type(users_features, items_features, config)
            result = _analyze_hybrid(model, evaluation, train, test)

        elif issubclass(model_type, AbstractModelCF):
            model = model_type(dataset.n_users, dataset.n_items, config)
            result = _analyze_model(model, evaluation, train, test)

        elif issubclass(model_type, AbstractModelMD):
            model = model_type(users_features, items_features, config)
            result = _analyze_model(model, evaluation, train, test)

        else:
            print('Invalid model_type: {}'.format(model_type))
            result = None

        results_single.append((name, result))

    return results_single


def print_models_single(results_single):
    for name, result in results_single:

        print('-------', name)

        # Check if from hybrid
        if isinstance(result, Tuple):
            print('Hybrid before xtrain:')
            print(result[0])
            print('Hybrid after xtrain:')
            print(result[1])
        else:
            print(result)
