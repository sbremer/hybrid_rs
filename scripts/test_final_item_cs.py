import script_chdir
from scripts.test_final import get_results
from evaluation.eval_script import print_results

results = get_results('ml100k', coldstart=True, cs_type='item')

print_results(results)

# ------- Hybrid_odefault
# RMSE: 1.0575 ± 0.0167  MAE: 0.8672 ± 0.0161  Prec@5: 0.6615 ± 0.0211  TopNAURC(k=100): 0.6083 ± 0.0217  runtime: 59.6996 ± 2.4362
# ------- Hybrid_odefault_SVDpp
# RMSE: 1.1324 ± 0.0147  MAE: 0.9390 ± 0.0151  Prec@5: 0.6035 ± 0.0240  TopNAURC(k=100): 0.5235 ± 0.0052  runtime: 16.3933 ± 0.6328
# ------- Hybrid_odefault_AttributeBiasAdvanced
# RMSE: 1.0553 ± 0.0166  MAE: 0.8646 ± 0.0160  Prec@5: 0.6609 ± 0.0218  TopNAURC(k=100): 0.5722 ± 0.0232  runtime: 10.1170 ± 1.2278
# ------- Hybrid_ndefault
# RMSE: 1.0594 ± 0.0158  MAE: 0.8669 ± 0.0156  Prec@5: 0.6605 ± 0.0228  TopNAURC(k=100): 0.7284 ± 0.0126  runtime: 43.9043 ± 1.1693
# ------- Hybrid_ndefault_SigmoidUserAsymFactoring
# RMSE: 1.3078 ± 0.0256  MAE: 1.0930 ± 0.0254  Prec@5: 0.6035 ± 0.0240  TopNAURC(k=100): 0.4062 ± 0.0043  runtime: 11.7675 ± 0.5185
# ------- Hybrid_ndefault_AttributeBiasAdvanced
# RMSE: 1.0479 ± 0.0147  MAE: 0.8568 ± 0.0141  Prec@5: 0.6632 ± 0.0209  TopNAURC(k=100): 0.5999 ± 0.0174  runtime: 6.0232 ± 0.6245
# ------- SigmoidItemAsymFactoring
# RMSE: 1.1402 ± 0.0145  MAE: 0.9456 ± 0.0151  Prec@5: 0.6035 ± 0.0240  TopNAURC(k=100): 0.5178 ± 0.0029  runtime: 15.8788 ± 0.2665
# ------- BiasEstimator
# RMSE: 1.0740 ± 0.0119  MAE: 0.8836 ± 0.0109  Prec@5: 0.6035 ± 0.0240  TopNAURC(k=100): 0.5621 ± 0.0035  runtime: 8.6185 ± 0.6899
# ------- SVD
# RMSE: 1.0700 ± 0.0115  MAE: 0.8840 ± 0.0110  Prec@5: 0.6035 ± 0.0240  TopNAURC(k=100): 0.5781 ± 0.0030  runtime: 8.7231 ± 0.1623
