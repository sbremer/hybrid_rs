from typing import Dict

# Local imports
from hybrid_model import evaluation_parting
from hybrid_model import evaluation_metrics

# Static stuff
parts = {'full': evaluation_parting.full}

metrics = {'rmse': evaluation_metrics.Rmse().calculate,
           'mae': evaluation_metrics.Mae().calculate,
           'ndcg@5': evaluation_metrics.Ndcg(5).calculate}


class EvaluationResult:
    def __init__(self):
        self.parts: Dict[str, EvaluationResultPart] = []


class EvaluationResultPart:
    def __init__(self):
        self.model_mf: EvaluationResultModel = EvaluationResultModel()
        self.model_cs: EvaluationResultModel = EvaluationResultModel()


class EvaluationResultModel:
    def __init__(self):
        self.results: Dict[str, float] = {}

    def __str__(self):
        s = ''
        for metric, result in self.results.items():
            s += '{}: {:.4f}\t'.format(metric, result)

        return s
