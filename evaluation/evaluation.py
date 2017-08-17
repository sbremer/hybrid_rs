from typing import Dict, List

import numpy as np

import hybrid_model
from evaluation import evaluation_metrics

metrics_default = [evaluation_metrics.Rmse(),
                   evaluation_metrics.Mae(),
                   evaluation_metrics.Precision(5),
                   evaluation_metrics.TopNAURC(100),
                   ]


class Evaluation:
    def __init__(self, metrics: List[evaluation_metrics.Metric] = metrics_default):

        self.metrics = metrics
        self.metric_names = [metric.__str__() for metric in metrics] + ['runtime']

    def evaluate_hybrid(self, model: 'hybrid_model.hybrid.HybridModel', x_train: List[np.ndarray], y_train: np.ndarray,
                        x_test: List[np.ndarray], y_test: np.ndarray) \
            -> 'EvaluationResultHybrid':
        result = EvaluationResultHybrid()
        result.cf = self.evaluate(model.model_cf, x_train, y_train, x_test, y_test)
        result.md = self.evaluate(model.model_md, x_train, y_train, x_test, y_test)

        return result

    def evaluate(self, model: 'hybrid_model.models.AbstractModel', x_train: List[np.ndarray], y_train: np.ndarray,
                 x_test: List[np.ndarray], y_test: np.ndarray) \
            -> 'EvaluationResult':
        result = EvaluationResult()

        # x_all = np.meshgrid(np.arange(model.n_users), np.arange(model.n_items))
        # x_all = [x_all[0].flatten(), x_all[1].flatten()]
        # y_all = np.reshape(model.predict(x_all), (model.n_users, model.n_items), 'F')
        #
        # y_pred = y_all[x_test[0], x_test[1]]
        y_pred = model.predict(x_test)

        for metric_name, metric in zip(self.metric_names, self.metrics):
            if issubclass(metric.__class__, evaluation_metrics.BasicMetric):
                result.results[metric_name] = metric.calculate(y_test, y_pred, x_test)
            elif issubclass(metric.__class__, evaluation_metrics.AdvancedMetric):
                result.results[metric_name] = metric.calculate(model, x_train, x_test, y_test, y_pred)
            else:
                raise TypeError('Unknown Metric!')

        return result

    def get_results_class(self):
        return EvaluationResults(self.metric_names)

    def get_results_hybrid_class(self):
        return EvaluationResultsHybrid(self.metric_names)


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
        self.results: Dict[str, float] = {}

    def __str__(self):
        s = ''
        for metric_name, result in self.results.items():
            s += '{}: {:.4f}  '.format(metric_name, result)

        return s

    def rmse(self):
        return self.results['rmse']


# === Multiple Evaluation Results (from Folds)
class EvaluationResultsHybrid:
    def __init__(self, metric_names: List[str] = [metric.__str__() for metric in metrics_default]):
        self.cf = EvaluationResults(metric_names)
        self.md = EvaluationResults(metric_names)

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
    def __init__(self, metric_names):
        self.results: Dict[str, List[float]] = dict((metric_name, []) for metric_name in metric_names)

    def __str__(self):
        s = ''
        for metric_name, result in self.results.items():
            mean = np.mean(result, 0)
            std = np.std(result, 0)
            if np.isscalar(mean) and np.isscalar(std):
                s += '{}: {:.4f} ± {:.4f}  '.format(metric_name, mean, std)
            else:
                s += '{}: {} ± {}  '.format(metric_name, repr(mean), repr(std))

        return s

    def add(self, result: EvaluationResult):
        for metric_name in self.results.keys():
            self.results[metric_name].append(result.results[metric_name])

    def mean(self, metric_name):
        return np.mean(self.results[metric_name], 0)

    def rmse(self):
        return self.mean('rmse')
