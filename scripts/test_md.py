import script_chdir
from evaluation.eval_script import evaluate_models_xval, EvalModel, print_results
from hybrid_model.dataset import get_dataset


# Get dataset
dataset = get_dataset('ml100k')

models = []

# Bias Baseline
from hybrid_model.models import BiasEstimator
model_type = BiasEstimator
config = {}
models.append(EvalModel(model_type.__name__, model_type, config))

# AttributeBiasAdvanced
from hybrid_model.models import AttributeBiasAdvanced
model_type = AttributeBiasAdvanced
config = {'reg_att_bias': 6.578729437598415e-07, 'reg_bias': 6.842025959062749e-07}
models.append(EvalModel(model_type.__name__, model_type, config))

# AttributeBiasLight
from hybrid_model.models import AttributeBiasLight
model_type = AttributeBiasLight
config = {'reg_bias': 0.00005, 'reg_att_bias': 0.0003}
models.append(EvalModel(model_type.__name__+'default', model_type, config))

config = {'reg_att_bias': 4.3518131605624814e-05, 'reg_bias': 6.936520853421938e-05}
models.append(EvalModel(model_type.__name__+'2:1', model_type, config))

config = {'reg_att_bias': 0.00014201148821999657, 'reg_bias': 0.0006657144990304895}
models.append(EvalModel(model_type.__name__+'user', model_type, config))

config = {'reg_att_bias': 4.156321415552967e-07, 'reg_bias': 0.01791415395580668}
models.append(EvalModel(model_type.__name__+'item', model_type, config))

print('User Cold-Start:')
results = evaluate_models_xval(dataset, models, coldstart=True, cs_type='user')
print_results(results)

print('Item Cold-Start:')
results = evaluate_models_xval(dataset, models, coldstart=True, cs_type='item')
print_results(results)

# User Cold-Start:
# ------- BiasEstimator
# RMSE: 1.0212 ± 0.0248  MAE: 0.8224 ± 0.0208  Prec@5: 0.5540 ± 0.0311  TopNAURC(k=100): 0.8165 ± 0.0107  runtime: 9.8669 ± 1.5382
# ------- AttributeBiasAdvanced
# RMSE: 1.0209 ± 0.0247  MAE: 0.8224 ± 0.0209  Prec@5: 0.5504 ± 0.0187  TopNAURC(k=100): 0.8097 ± 0.0114  runtime: 9.9565 ± 0.3252
# ------- AttributeBiasLightdefault
# RMSE: 1.0204 ± 0.0246  MAE: 0.8235 ± 0.0210  Prec@5: 0.5540 ± 0.0278  TopNAURC(k=100): 0.8206 ± 0.0105  runtime: 10.1875 ± 0.5145
# ------- AttributeBiasLight2:1
# RMSE: 1.0242 ± 0.0238  MAE: 0.8293 ± 0.0209  Prec@5: 0.5574 ± 0.0199  TopNAURC(k=100): 0.8278 ± 0.0087  runtime: 7.5154 ± 0.2977
# ------- AttributeBiasLightuser
# RMSE: 1.0247 ± 0.0231  MAE: 0.8269 ± 0.0194  Prec@5: 0.5544 ± 0.0212  TopNAURC(k=100): 0.8305 ± 0.0088  runtime: 8.7820 ± 0.1410
# ------- AttributeBiasLightitem
# RMSE: 1.0336 ± 0.0233  MAE: 0.8347 ± 0.0197  Prec@5: 0.5379 ± 0.0179  TopNAURC(k=100): 0.8143 ± 0.0098  runtime: 9.0803 ± 0.3275
# Item Cold-Start:
# ------- BiasEstimator
# RMSE: 1.0758 ± 0.0153  MAE: 0.8857 ± 0.0162  Prec@5: 0.6065 ± 0.0303  TopNAURC(k=100): 0.5608 ± 0.0058  runtime: 9.6690 ± 0.6200
# ------- AttributeBiasAdvanced
# RMSE: 1.0506 ± 0.0155  MAE: 0.8593 ± 0.0155  Prec@5: 0.6666 ± 0.0305  TopNAURC(k=100): 0.5949 ± 0.0171  runtime: 9.5325 ± 0.1435
# ------- AttributeBiasLightdefault
# RMSE: 1.0710 ± 0.0142  MAE: 0.8809 ± 0.0140  Prec@5: 0.6410 ± 0.0244  TopNAURC(k=100): 0.5677 ± 0.0059  runtime: 10.0029 ± 0.8026
# ------- AttributeBiasLight2:1
# RMSE: 1.0629 ± 0.0140  MAE: 0.8737 ± 0.0136  Prec@5: 0.6475 ± 0.0249  TopNAURC(k=100): 0.5715 ± 0.0066  runtime: 7.3104 ± 0.2125
# ------- AttributeBiasLightuser
# RMSE: 1.0600 ± 0.0127  MAE: 0.8723 ± 0.0124  Prec@5: 0.6462 ± 0.0261  TopNAURC(k=100): 0.5632 ± 0.0087  runtime: 8.4840 ± 0.7145
# ------- AttributeBiasLightitem
# RMSE: 1.0549 ± 0.0127  MAE: 0.8639 ± 0.0125  Prec@5: 0.6511 ± 0.0276  TopNAURC(k=100): 0.5850 ± 0.0079  runtime: 9.1323 ± 0.2750

