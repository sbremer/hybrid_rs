from typing import Dict, List

import numpy as np

import hybrid_model
from evaluation import evaluation_metrics
from evaluation import evaluation_parting

metrics_rmse = {'rmse': evaluation_metrics.Rmse()}

metrics_rmse_prec = {'rmse': evaluation_metrics.Rmse(),
                     'prec@5': evaluation_metrics.Precision(5)}

metrics_all = {'rmse': evaluation_metrics.Rmse(),
               'mae': evaluation_metrics.Mae(),
               'prec@5': evaluation_metrics.Precision(5),
               'ndcg@5': evaluation_metrics.Ndcg(5)}

parting_full = {'full': evaluation_parting.Full()}


def get_parting_all(n_bins):
    parting = {'full': evaluation_parting.Full()}

    parting.update({'user_{}'.format(i+1):
                        evaluation_parting.BinningUser(n_bins, i) for i in range(n_bins)})
    parting.update({'item_{}'.format(i+1):
                        evaluation_parting.BinningItem(n_bins, i) for i in range(n_bins)})

    return parting


class Evaluation:
    def __init__(self,
                 metrics: Dict[str, evaluation_metrics.Metric] = metrics_rmse_prec,
                 parts: Dict[str, evaluation_parting.Parting] = parting_full):

        self.metrics = metrics
        self.parts = parts

    def evaluate_hybrid(self, model: 'hybrid_model.hybrid.HybridModel', x_test: List[np.ndarray], y_test: np.ndarray) \
            -> 'EvaluationResultHybrid':
        result = EvaluationResultHybrid()
        result.cf = self.evaluate(model.model_cf, x_test, y_test)
        result.md = self.evaluate(model.model_md, x_test, y_test)

        return result

    def evaluate(self, model: 'hybrid_model.models.AbstractModel', x_test: List[np.ndarray], y_test: np.ndarray) \
            -> 'EvaluationResult':
        result = EvaluationResult()

        for part, parting in self.parts.items():
            x_test_part, y_test_part = parting.part(x_test, y_test)
            result_part = self.evaluate_part(model, x_test_part, y_test_part)
            result.parts[part] = result_part

        return result

    def evaluate_part(self, model: 'hybrid_model.models.AbstractModel', x_test: List[np.ndarray], y_test: np.ndarray) \
            -> 'EvaluationResultPart':
        result = EvaluationResultPart()
        y_pred = model.predict(x_test)

        for measure, metric in self.metrics.items():
            result.results[measure] = metric.calculate(y_test, y_pred, x_test)

        return result

    def get_results_class(self):
        return EvaluationResults(self.metrics, self.parts)

    def get_results_hybrid_class(self):
        return EvaluationResultsHybrid(self.metrics, self.parts)

    def update_parts(self, user_dist, item_dist):
        for part in self.parts.keys():
            self.parts[part].update(user_dist, item_dist)


# === Single Evaluation Results
class EvaluationResultHybrid:
    def __init__(self):
        self.cf = EvaluationResult()
        self.md = EvaluationResult()

    def __str__(self):
        s = 'CF:\n'
        s += str(self.cf)
        s += 'MD:\n'
        s += str(self.md)

        return s


class EvaluationResult:
    def __init__(self):
        self.parts: Dict[str, EvaluationResultPart] = {}

    def __str__(self):
        s = ''
        for part, result in self.parts.items():
            s += '=== Part {}\n'.format(part)
            s += str(result)
            s += '\n'

        return s

    def rmse(self):
        return self.parts['full'].results['rmse']


class EvaluationResultPart:
    def __init__(self):
        self.results: Dict[str, float] = {}

    def __str__(self):
        s = ''
        for metric, result in self.results.items():
            s += '{}: {:.4f}  '.format(metric, result)

        return s


# === Multiple Evaluation Results (from Folds)
class EvaluationResultsHybrid:
    def __init__(self, metrics: List[str] = metrics_rmse.keys(), parts: List[str] = parting_full.keys()):
        self.cf = EvaluationResults(metrics, parts)
        self.md = EvaluationResults(metrics, parts)

    def add(self, result: EvaluationResultHybrid):
        self.cf.add(result.cf)
        self.md.add(result.md)

    def __str__(self):
        s = 'CF:\n'
        s += str(self.cf)
        s += 'MD:\n'
        s += str(self.md)

        return s

    def mean_rmse_cf(self):
        rmse = self.cf.rmse()
        return rmse

    def mean_rmse_md(self):
        """
        Custom hacky function for Gridsearch
        """
        rmse = self.md.rmse()
        return rmse


class EvaluationResults:
    def __init__(self, metrics: List[str] = metrics_rmse.keys(), parts: List[str] = parting_full.keys()):
        self.parts: Dict[str, EvaluationResultsPart] = dict((key, EvaluationResultsPart(metrics)) for key in parts)

    def add(self, result: EvaluationResult):
        for part in self.parts.keys():
            self.parts[part].add(result.parts[part])

    def __str__(self):
        s = ''
        for part, result in self.parts.items():
            s += '=== Part {}\n'.format(part)
            s += str(result)
            s += '\n'

        return s

    def rmse(self):
        return self.parts['full'].mean('rmse')


class EvaluationResultsPart:
    def __init__(self, metrics):
        self.results: Dict[str, List[float]] = dict((key, []) for key in metrics)

    def __str__(self):
        s = ''
        for metric, result in self.results.items():
            mean = np.mean(result)
            std = np.std(result)
            s += '{}: {:.4f} ± {:.4f}  '.format(metric, mean, std)

        return s

    def add(self, result: EvaluationResultPart):
        for metric in self.results.keys():
            self.results[metric].append(result.results[metric])

    def mean(self, metric):
        return np.mean(self.results[metric])
