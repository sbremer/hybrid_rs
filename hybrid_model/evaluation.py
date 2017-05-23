import numpy as np
from typing import Dict, List

# Local imports
import hybrid_model
from hybrid_model import evaluation_parting
from hybrid_model import evaluation_metrics

metrics_rmse = {'rmse': evaluation_metrics.Rmse()}

metrics_all = {'rmse': evaluation_metrics.Rmse(),
               'mae': evaluation_metrics.Mae(),
               'ndcg@5': evaluation_metrics.Ndcg(5)}

parting_full = {'full': evaluation_parting.Full()}


def get_parting_all(n_bins, user_dist, item_dist):
    parting = {'full': evaluation_parting.Full()}

    parting.update({'user_{}'.format(i+1):
                        evaluation_parting.BinningUser(n_bins, i, user_dist, item_dist) for i in range(n_bins)})
    parting.update({'item_{}'.format(i+1):
                        evaluation_parting.BinningItem(n_bins, i, user_dist, item_dist) for i in range(n_bins)})

    return parting


class Evaluation:
    def __init__(self,
                 metrics: Dict[str, evaluation_metrics.Metric] = metrics_rmse,
                 parts: Dict[str, evaluation_parting.Parting] = parting_full):

        self.metrics = metrics
        self.parts = parts

    def evaluate_hybrid(self, model: 'hybrid_model.hybrid.HybridModel', x_test: List[np.ndarray], y_test: np.ndarray) \
            -> 'EvaluationResultHybrid':
        result = EvaluationResultHybrid()
        result.mf = self.evaluate(model.model_mf, x_test, y_test)
        result.cs = self.evaluate(model.model_cs, x_test, y_test)

        return result

    def evaluate(self, model: 'hybrid_model.models.AbstractKerasModel', x_test: List[np.ndarray], y_test: np.ndarray) \
            -> 'EvaluationResult':
        result = EvaluationResult()

        for part, parting in self.parts.items():
            x_test_part, y_test_part = parting.part(x_test, y_test)
            result_part = self.evaluate_part(model, x_test_part, y_test_part)
            result.parts[part] = result_part

        return result

    def evaluate_part(self, model: 'hybrid_model.models.AbstractKerasModel', x_test: List[np.ndarray], y_test: np.ndarray) \
            -> 'EvaluationResultPart':
        result = EvaluationResultPart()
        y_pred = model.predict(x_test)

        for measure, metric in self.metrics.items():
            result.results[measure] = metric.calculate(y_test, y_pred, x_test)

        return result


# === Single Evaluation Results
class EvaluationResultHybrid:
    def __init__(self):
        self.mf = EvaluationResult()
        self.cs = EvaluationResult()

    def __str__(self):
        s = 'MF:\n'
        s += str(self.mf)
        s += 'CS:\n'
        s += str(self.cs)

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
    def __init__(self, metrics: List[str], parts: List[str]):
        self.mf = EvaluationResults(metrics, parts)
        self.cs = EvaluationResults(metrics, parts)

    def add(self, result: EvaluationResultHybrid):
        self.mf.add(result.mf)
        self.cs.add(result.cs)

    def __str__(self):
        s = 'MF:\n'
        s += str(self.mf)
        s += 'CS:\n'
        s += str(self.cs)

        return s

class EvaluationResults:
    def __init__(self, metrics: List[str], parts: List[str]):
        self.parts: Dict[str, EvaluationResultsPart] = dict((key, EvaluationResultsPart(metrics)) for key in parts)

    def add(self, result: EvaluationResult):
        for part in self.parts.keys():
            self.parts[part].add(result.parts[part])

    def __str__(self):
        s = 'Combined Results: \n'
        for part, result in self.parts.items():
            s += '=== Part {}\n'.format(part)
            s += str(result)
            s += '\n'

        return s

    def mean_rmse_mf(self):
        """
        Custom hacky function for Gridsearch
        """
        rmses = self.parts['full'].results['rmse']
        return np.mean(rmses)

    def mean_rmse_cs(self):
        """
        Custom hacky function for Gridsearch
        """
        rmses = self.parts['full'].results['rmse']
        return np.mean(rmses)


class EvaluationResultsPart:
    def __init__(self, metrics):
        self.results: Dict[str, List[float]] = dict((key, []) for key in metrics)
        self.metrics = metrics

    def __str__(self):
        s = ''
        for metric, result in self.results.items():
            mean = np.mean(result)
            std = np.std(result)
            s += '{}: {:.4f} ± {:.4f}  '.format(metric, mean, std)

        return s

    def add(self, result: EvaluationResultPart):
        for metric in self.metrics.keys():
            self.results[metric].append(result.results[metric])

    def mean(self, metric):
        return np.mean(self.results[metric])
