import script_chdir
from scripts.test_final import get_results
from evaluation.eval_script import print_results

results = get_results('ml100k', coldstart=False)

print_results(results)

# ------- Hybrid_odefault
# RMSE: 0.8956 ± 0.0027  MAE: 0.7070 ± 0.0031  Prec@5: 0.7828 ± 0.0028  TopNAURC(k=100): 0.8501 ± 0.0027  runtime: 57.6225 ± 3.5537
# ------- Hybrid_odefault_SVDpp
# RMSE: 0.8999 ± 0.0033  MAE: 0.7116 ± 0.0036  Prec@5: 0.7787 ± 0.0024  TopNAURC(k=100): 0.8729 ± 0.0022  runtime: 16.9510 ± 1.0995
# ------- Hybrid_odefault_AttributeBiasAdvanced
# RMSE: 0.9267 ± 0.0047  MAE: 0.7331 ± 0.0040  Prec@5: 0.7581 ± 0.0022  TopNAURC(k=100): 0.8196 ± 0.0024  runtime: 8.8394 ± 1.2390
# ------- Hybrid_ndefault
# RMSE: 0.8938 ± 0.0037  MAE: 0.7053 ± 0.0034  Prec@5: 0.7851 ± 0.0026  TopNAURC(k=100): 0.8988 ± 0.0030  runtime: 41.7555 ± 1.1406
# ------- Hybrid_ndefault_SigmoidUserAsymFactoring
# RMSE: 0.8951 ± 0.0039  MAE: 0.7070 ± 0.0035  Prec@5: 0.7825 ± 0.0031  TopNAURC(k=100): 0.9036 ± 0.0019  runtime: 11.3529 ± 0.8660
# ------- Hybrid_ndefault_AttributeBiasAdvanced
# RMSE: 0.9290 ± 0.0042  MAE: 0.7342 ± 0.0034  Prec@5: 0.7577 ± 0.0023  TopNAURC(k=100): 0.8197 ± 0.0015  runtime: 5.9371 ± 0.2633
# ------- SigmoidItemAsymFactoring
# RMSE: 0.9021 ± 0.0034  MAE: 0.7135 ± 0.0032  Prec@5: 0.7771 ± 0.0033  TopNAURC(k=100): 0.8712 ± 0.0023  runtime: 16.5498 ± 1.4323
# ------- BiasEstimator
# RMSE: 0.9420 ± 0.0050  MAE: 0.7459 ± 0.0043  Prec@5: 0.7482 ± 0.0021  TopNAURC(k=100): 0.8225 ± 0.0031  runtime: 8.7634 ± 1.2045
# ------- SVD
# RMSE: 0.9244 ± 0.0046  MAE: 0.7330 ± 0.0038  Prec@5: 0.7645 ± 0.0022  TopNAURC(k=100): 0.8511 ± 0.0023  runtime: 9.3913 ± 0.6819

# ------- Hybrid_odefault
# RMSE: 0.8958 ± 0.0026  MAE: 0.7073 ± 0.0030  Prec@5: 0.7818 ± 0.0020  TopNAURC(k=100): 0.8489 ± 0.0021  runtime: 57.9287 ± 1.0305
# ------- Hybrid_odefault_SVDpp
# RMSE: 0.8996 ± 0.0024  MAE: 0.7112 ± 0.0023  Prec@5: 0.7771 ± 0.0023  TopNAURC(k=100): 0.8724 ± 0.0032  runtime: 17.9672 ± 0.7607
# ------- Hybrid_odefault_AttributeBiasAdvanced
# RMSE: 0.9266 ± 0.0041  MAE: 0.7328 ± 0.0035  Prec@5: 0.7587 ± 0.0025  TopNAURC(k=100): 0.8200 ± 0.0019  runtime: 9.1443 ± 0.3485
# ------- Hybrid_ndefault
# RMSE: 0.8942 ± 0.0028  MAE: 0.7053 ± 0.0026  Prec@5: 0.7830 ± 0.0016  TopNAURC(k=100): 0.8996 ± 0.0009  runtime: 40.9590 ± 1.5013
# ------- Hybrid_ndefault_SigmoidUserAsymFactoring
# RMSE: 0.8942 ± 0.0029  MAE: 0.7058 ± 0.0023  Prec@5: 0.7840 ± 0.0016  TopNAURC(k=100): 0.9036 ± 0.0013  runtime: 12.8925 ± 1.2344
# ------- Hybrid_ndefault_AttributeBiasLight
# RMSE: 0.9294 ± 0.0045  MAE: 0.7362 ± 0.0039  Prec@5: 0.7596 ± 0.0023  TopNAURC(k=100): 0.8366 ± 0.0017  runtime: 4.2876 ± 0.2542
# ------- SigmoidItemAsymFactoring
# RMSE: 0.9025 ± 0.0025  MAE: 0.7145 ± 0.0026  Prec@5: 0.7774 ± 0.0030  TopNAURC(k=100): 0.8701 ± 0.0018  runtime: 16.6412 ± 0.7794
# ------- BiasEstimator
# RMSE: 0.9416 ± 0.0046  MAE: 0.7454 ± 0.0037  Prec@5: 0.7482 ± 0.0026  TopNAURC(k=100): 0.8225 ± 0.0032  runtime: 10.1067 ± 1.5695
# ------- SVD
# RMSE: 0.9248 ± 0.0047  MAE: 0.7330 ± 0.0035  Prec@5: 0.7639 ± 0.0023  TopNAURC(k=100): 0.8508 ± 0.0020  runtime: 9.5241 ± 0.2553
