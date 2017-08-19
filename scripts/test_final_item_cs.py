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

# ------- Hybrid_odefault
# RMSE: 1.0590 ± 0.0183  MAE: 0.8686 ± 0.0175  Prec@5: 0.6611 ± 0.0210  TopNAURC(k=100): 0.6055 ± 0.0223  runtime: 58.4032 ± 1.5156
# ------- Hybrid_odefault_SVDpp
# RMSE: 1.1325 ± 0.0132  MAE: 0.9389 ± 0.0141  Prec@5: 0.6035 ± 0.0240  TopNAURC(k=100): 0.5247 ± 0.0041  runtime: 16.9710 ± 0.6437
# ------- Hybrid_odefault_AttributeBiasAdvanced
# RMSE: 1.0552 ± 0.0164  MAE: 0.8646 ± 0.0157  Prec@5: 0.6606 ± 0.0216  TopNAURC(k=100): 0.5713 ± 0.0231  runtime: 9.8551 ± 1.1968
# ------- Hybrid_ndefault
# RMSE: 1.0732 ± 0.0150  MAE: 0.8829 ± 0.0149  Prec@5: 0.6459 ± 0.0229  TopNAURC(k=100): 0.7003 ± 0.0068  runtime: 38.2648 ± 0.3375
# ------- Hybrid_ndefault_SigmoidUserAsymFactoring
# RMSE: 1.3078 ± 0.0165  MAE: 1.0929 ± 0.0179  Prec@5: 0.6035 ± 0.0240  TopNAURC(k=100): 0.4063 ± 0.0044  runtime: 11.2220 ± 0.3622
# ------- Hybrid_ndefault_AttributeBiasLight
# RMSE: 1.0618 ± 0.0140  MAE: 0.8733 ± 0.0136  Prec@5: 0.6438 ± 0.0237  TopNAURC(k=100): 0.5726 ± 0.0083  runtime: 3.7552 ± 0.1044
# ------- SigmoidItemAsymFactoring
# RMSE: 1.1403 ± 0.0138  MAE: 0.9457 ± 0.0146  Prec@5: 0.6035 ± 0.0240  TopNAURC(k=100): 0.5174 ± 0.0022  runtime: 16.1313 ± 1.0070
# ------- BiasEstimator
# RMSE: 1.0745 ± 0.0110  MAE: 0.8841 ± 0.0102  Prec@5: 0.6035 ± 0.0240  TopNAURC(k=100): 0.5608 ± 0.0019  runtime: 8.3132 ± 0.9333
# ------- SVD
# RMSE: 1.0690 ± 0.0122  MAE: 0.8830 ± 0.0116  Prec@5: 0.6035 ± 0.0240  TopNAURC(k=100): 0.5775 ± 0.0024  runtime: 8.5058 ± 0.1678
