from evaluation.eval_script import evaluate_models_xval, EvalModel
from hybrid_model.dataset import get_dataset


def get_results(dataset_name, coldstart, cs_type='none', n_entries=0):

    # Get dataset
    dataset = get_dataset(dataset_name)

    models_test = []

    # Hybrid Model
    from hybrid_model.hybrid import HybridModel
    model_type = HybridModel

    from hybrid_model.config import hybrid_config
    config = hybrid_config

    models_test.append(EvalModel('Hybrid_odefault', model_type, config))

    from hybrid_model.config import hybrid_config_new2
    config = hybrid_config_new2

    models_test.append(EvalModel('Hybrid_ndefault', model_type, config))

    # SigmoidItemAsymFactoring
    from hybrid_model.models import SigmoidItemAsymFactoring
    model_type = SigmoidItemAsymFactoring
    config = {}
    models_test.append(EvalModel(model_type.__name__, model_type, config))

    # Bias Baseline
    from hybrid_model.models import BiasEstimator
    model_type = BiasEstimator
    config = {}
    models_test.append(EvalModel(model_type.__name__, model_type, config))

    # SVD
    from hybrid_model.models import SVD
    model_type = SVD
    config = {}
    models_test.append(EvalModel(model_type.__name__, model_type, config))

    results = evaluate_models_xval(dataset, models_test, coldstart=coldstart, cs_type=cs_type, n_entries=n_entries)

    return results