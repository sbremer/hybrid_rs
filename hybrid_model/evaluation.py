import numpy as np
from typing import Dict, List

# Local imports
from hybrid_model import evaluation_parting
from hybrid_model import evaluation_metrics

# Static stuff
parts = {'full': evaluation_parting.full}

metrics = {'rmse': evaluation_metrics.Rmse().calculate,
           'mae': evaluation_metrics.Mae().calculate,
           'ndcg@5': evaluation_metrics.Ndcg(5).calculate}


# === Single Evaluation Results
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
        self.model_mf: EvaluationResultModel = EvaluationResultModel()
        self.model_cs: EvaluationResultModel = EvaluationResultModel()

    def __str__(self):
        return str(self.model_mf) + '\n' + str(self.model_cs)


class EvaluationResultModel:
    def __init__(self):
        self.results: Dict[str, float] = {}

    def __str__(self):
        s = ''
        for metric, result in self.results.items():
            s += '{}: {:.4f}  '.format(metric, result)

        return s


# === Multiple Evaluation Results (from Folds)
class EvaluationResults:
    def __init__(self):
        self.parts: Dict[str, EvaluationResultsPart] = dict((key, EvaluationResultsPart) for key in parts.keys())

    def add(self, result: EvaluationResult):
        for part in parts.keys():
            self.parts[part].add(result.parts[part])

    def __str__(self):
        s = ''
        for part, result in self.parts.items():
            s += '=== Part {}\n'.format(part)
            s += str(result)
            s += '\n'

    def mean_rmse_mf(self):
        """
        Custom hacky function for Gridsearch
        """
        rmses = self.parts['full'].model_mf.results['rmse']
        return np.mean(rmses)

    def mean_rmse_cs(self):
        """
        Custom hacky function for Gridsearch
        """
        rmses = self.parts['full'].model_cs.results['rmse']
        return np.mean(rmses)


class EvaluationResultsPart:
    def __init__(self):
        self.model_mf: EvaluationResultsModel = EvaluationResultsModel()
        self.model_cs: EvaluationResultsModel = EvaluationResultsModel()

    def __str__(self):
        return str(self.model_mf) + '\n' + str(self.model_cs)

    def add(self, result: EvaluationResultPart):
        self.model_mf.add(result.model_mf)
        self.model_cs.add(result.model_cs)


class EvaluationResultsModel:
    def __init__(self):
        self.results: Dict[str, List[float]] = dict((key, []) for key in metrics.keys())

    def __str__(self):
        s = ''
        for metric, result in self.results.items():
            mean = np.mean(result)
            std = np.std(result)
            s += '{}: {:.4f} ± {:.4f}  '.format(metric, mean, std)

        return s

    def add(self, result: EvaluationResultModel):
        for metric in metrics.keys():
            self.results[metric].append(result.results[metric])