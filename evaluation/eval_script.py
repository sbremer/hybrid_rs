from typing import NamedTuple, List, Dict, Tuple, Type

import numpy as np

np.random.seed(0)

# Local imports
from hybrid_model.hybrid import HybridModel
from evaluation.evaluation import Evaluation, EvaluationResult, EvaluationResultHybrid
from hybrid_model.models import AbstractModel, AbstractModelCF, AbstractModelMD
from hybrid_model.dataset import Dataset
from util import kfold, timing


class EvalModel(NamedTuple):
    name: str
    model_type: Type[AbstractModel]
    config: Dict


def _analyze_hybrid(model: HybridModel, evaluater: Evaluation, train, test)\
        -> Tuple[EvaluationResultHybrid, EvaluationResultHybrid]:

    with timing.Timer() as t1:
        model.fit_init(*train)

    result_before_x = evaluater.evaluate_hybrid(model, *train, *test)

    with timing.Timer() as t2:
        model.fit_cross()

    result_after_x = evaluater.evaluate_hybrid(model, *train, *test)

    return result_before_x, result_after_x


def _analyze_hybrid_as_model(model: HybridModel, evaluater: Evaluation, train, test)\
        -> Tuple[EvaluationResult, EvaluationResult, EvaluationResult]:

    with timing.Timer() as t1:
        model.fit_init(*train)

    result_before_x = evaluater.evaluate_hybrid(model, *train, *test)

    with timing.Timer() as t2:
        model.fit_cross()

    result_hybrid = evaluater.evaluate(model, *train, *test)
    result_hybrid.results['runtime'] = t1.interval + t2.interval

    return result_hybrid, result_before_x.cf, result_before_x.md


def _analyze_model(model, evaluation: Evaluation, train, test)\
        -> EvaluationResult:

    with timing.Timer() as t:
        model.fit(*train)

    result = evaluation.evaluate(model, *train, *test)
    result.results['runtime'] = t.interval

    return result


def evaluate_models_xval(dataset: Dataset, models: List[EvalModel], coldstart=False, cs_type='user', n_entries=0,
                         evaluater=None, n_fold=5, repeat=1):
    (inds_u, inds_i, y, users_features, items_features) = dataset.data

    folds = []

    for _ in range(repeat):
        if coldstart and cs_type == 'user':
            if n_entries == 0:
                fold = kfold.kfold_entries(n_fold, inds_u)
            else:
                fold = kfold.kfold_entries_plus(n_fold, inds_u, n_entries)
        elif coldstart and cs_type == 'item':
            if n_entries == 0:
                fold = kfold.kfold_entries(n_fold, inds_i)
            else:
                fold = kfold.kfold_entries_plus(n_fold, inds_i, n_entries)
        elif coldstart:
            raise ValueError('unknown cs_type')
        else:
            fold = kfold.kfold(n_fold, inds_u)

        folds.extend(list(fold))

    if evaluater is None:
        evaluater = Evaluation()

    # Create results list
    results = {}
    for name, model_type, config in models:
        if issubclass(model_type, AbstractModel):
            results[name] = evaluater.get_results_class()

        elif issubclass(model_type, HybridModel):
            results[name] = evaluater.get_results_class()
            results[name + '_' + config.model_type_cf.__name__] = evaluater.get_results_class()
            results[name + '_' + config.model_type_md.__name__] = evaluater.get_results_class()

        else:
            raise TypeError('Invalid model_type')

    for xval_train, xval_test in folds:

        # Dataset training
        inds_u_train = inds_u[xval_train]
        inds_i_train = inds_i[xval_train]
        y_train = y[xval_train]

        # Dataset testing
        inds_u_test = inds_u[xval_test]
        inds_i_test = inds_i[xval_test]
        y_test = y[xval_test]

        train = ([inds_u_train, inds_i_train], y_train)
        test = ([inds_u_test, inds_i_test], y_test)

        for name, model_type, config in models:

            if issubclass(model_type, AbstractModelCF):
                model = model_type(dataset.n_users, dataset.n_items, config)
                result = _analyze_model(model, evaluater, train, test)
                results[name].add(result)

            elif issubclass(model_type, AbstractModelMD):
                model = model_type(users_features, items_features, config)
                result = _analyze_model(model, evaluater, train, test)
                results[name].add(result)

            elif issubclass(model_type, HybridModel):
                model = model_type(users_features, items_features, config)
                result_hybrid, result_cf, result_md = _analyze_hybrid_as_model(model, evaluater, train, test)

                results[name].add(result_hybrid)
                results[name + '_' + config.model_type_cf.__name__].add(result_cf)
                results[name + '_' + config.model_type_md.__name__].add(result_md)

    return results


def evaluate_models_single(dataset: Dataset, models: List[EvalModel], coldstart=False, cs_type='user', n_entries=0,
                           evaluater=None, n_fold=5):
    (inds_u, inds_i, y, users_features, items_features) = dataset.data

    if coldstart and cs_type == 'user':
        if n_entries == 0:
            fold = kfold.kfold_entries(n_fold, inds_u)
        else:
            fold = kfold.kfold_entries_plus(n_fold, inds_u, n_entries)
    elif coldstart and cs_type == 'item':
        if n_entries == 0:
            fold = kfold.kfold_entries(n_fold, inds_i)
        else:
            fold = kfold.kfold_entries_plus(n_fold, inds_i, n_entries)
    elif coldstart:
        raise ValueError('unknown cs_type')
    else:
        fold = kfold.kfold(n_fold, inds_u)

    fold = list(fold)

    xval_train, xval_test = fold[2]

    # Dataset training
    inds_u_train = inds_u[xval_train]
    inds_i_train = inds_i[xval_train]
    y_train = y[xval_train]

    # Dataset testing
    inds_u_test = inds_u[xval_test]
    inds_i_test = inds_i[xval_test]
    y_test = y[xval_test]

    if evaluater is None:
        evaluater = Evaluation()

    train = ([inds_u_train, inds_i_train], y_train)
    test = ([inds_u_test, inds_i_test], y_test)

    results = {}

    for name, model_type, config in models:

        if issubclass(model_type, AbstractModelCF):
            model = model_type(dataset.n_users, dataset.n_items, config)
            results[name] = _analyze_model(model, evaluater, train, test)

        elif issubclass(model_type, AbstractModelMD):
            model = model_type(users_features, items_features, config)
            results[name] = _analyze_model(model, evaluater, train, test)

        elif issubclass(model_type, HybridModel):
            model = model_type(users_features, items_features, config)
            results[name],\
            results[name + '_' + config.model_type_cf.__name__],\
            results[name + '_' + config.model_type_md.__name__] = \
                _analyze_hybrid_as_model(model, evaluater, train, test)

        else:
            raise TypeError('Invalid model_type')

    return results


def evaluate_hybrid_xval(dataset: Dataset, config, coldstart=False, cs_type='user', n_entries=0,
                         evaluater=None, n_fold=5, repeat=1):
    (inds_u, inds_i, y, users_features, items_features) = dataset.data

    folds = []

    for _ in range(repeat):
        if coldstart and cs_type == 'user':
            if n_entries == 0:
                fold = kfold.kfold_entries(n_fold, inds_u)
            else:
                fold = kfold.kfold_entries_plus(n_fold, inds_u, n_entries)
        elif coldstart and cs_type == 'item':
            if n_entries == 0:
                fold = kfold.kfold_entries(n_fold, inds_i)
            else:
                fold = kfold.kfold_entries_plus(n_fold, inds_i, n_entries)
        elif coldstart:
            raise ValueError('unknown cs_type')
        else:
            fold = kfold.kfold(n_fold, inds_u)

        folds.extend(list(fold))

    if evaluater is None:
        evaluater = Evaluation()

    # Create results list
    results_before = evaluater.get_results_hybrid_class()
    results_after = evaluater.get_results_hybrid_class()

    for xval_train, xval_test in folds:

        # Dataset training
        inds_u_train = inds_u[xval_train]
        inds_i_train = inds_i[xval_train]
        y_train = y[xval_train]
        n_train = len(y_train)

        # Dataset testing
        inds_u_test = inds_u[xval_test]
        inds_i_test = inds_i[xval_test]
        y_test = y[xval_test]

        train = ([inds_u_train, inds_i_train], y_train)
        test = ([inds_u_test, inds_i_test], y_test)

        model = HybridModel(users_features, items_features, config)

        result_before, result_after = _analyze_hybrid(model, evaluater, train, test)

        results_before.add(result_before)
        results_after.add(result_after)

    return results_before, results_after


def evaluate_hybrid_single(dataset: Dataset, config, coldstart=False, cs_type='user', n_entries=0,
                           evaluater=None, n_fold=5):
    (inds_u, inds_i, y, users_features, items_features) = dataset.data

    if coldstart and cs_type == 'user':
        if n_entries == 0:
            fold = kfold.kfold_entries(n_fold, inds_u)
        else:
            fold = kfold.kfold_entries_plus(n_fold, inds_u, n_entries)
    elif coldstart and cs_type == 'item':
        if n_entries == 0:
            fold = kfold.kfold_entries(n_fold, inds_i)
        else:
            fold = kfold.kfold_entries_plus(n_fold, inds_i, n_entries)
    elif coldstart:
        raise ValueError('unknown cs_type')
    else:
        fold = kfold.kfold(n_fold, inds_u)

    fold = list(fold)

    xval_train, xval_test = fold[2]

    if evaluater is None:
        evaluater = Evaluation()

    # Dataset training
    inds_u_train = inds_u[xval_train]
    inds_i_train = inds_i[xval_train]
    y_train = y[xval_train]

    # Dataset testing
    inds_u_test = inds_u[xval_test]
    inds_i_test = inds_i[xval_test]
    y_test = y[xval_test]

    train = ([inds_u_train, inds_i_train], y_train)
    test = ([inds_u_test, inds_i_test], y_test)

    model = HybridModel(users_features, items_features, config)

    result_before, result_after = _analyze_hybrid(model, evaluater, train, test)

    return result_before, result_after


def print_hybrid_results(results: Tuple):
    print('Before Cross-Training:')
    print(results[0])
    print('After Cross-Training:')
    print(results[1])


def print_results(results: Dict):
    for name, result in results.items():
        print('-------', name)
        print(result)
