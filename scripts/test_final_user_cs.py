import script_chdir
from scripts.test_final import get_results
from evaluation.eval_script import print_results

results = get_results('ml100k', coldstart=True, cs_type='user')

print_results(results)

# ------- Hybrid_odefault
# RMSE: 1.0169 ± 0.0256  MAE: 0.8173 ± 0.0219  Prec@5: 0.5584 ± 0.0191  TopNAURC(k=100): 0.8110 ± 0.0112  runtime: 60.0910 ± 2.0622
# ------- Hybrid_odefault_SVDpp
# RMSE: 1.0872 ± 0.0205  MAE: 0.9048 ± 0.0187  Prec@5: 0.5319 ± 0.0307  TopNAURC(k=100): 0.7979 ± 0.0124  runtime: 15.2592 ± 0.8154
# ------- Hybrid_odefault_AttributeBiasAdvanced
# RMSE: 1.0179 ± 0.0254  MAE: 0.8181 ± 0.0219  Prec@5: 0.5576 ± 0.0193  TopNAURC(k=100): 0.8090 ± 0.0121  runtime: 9.6782 ± 1.7367
# ------- Hybrid_ndefault
# RMSE: 1.0288 ± 0.0238  MAE: 0.8331 ± 0.0206  Prec@5: 0.5427 ± 0.0105  TopNAURC(k=100): 0.8200 ± 0.0105  runtime: 44.1241 ± 0.7063
# ------- Hybrid_ndefault_SigmoidUserAsymFactoring
# RMSE: 1.2111 ± 0.0165  MAE: 0.9906 ± 0.0176  Prec@5: 0.4816 ± 0.0328  TopNAURC(k=100): 0.5672 ± 0.0074  runtime: 11.7388 ± 0.2794
# ------- Hybrid_ndefault_AttributeBiasAdvanced
# RMSE: 1.0224 ± 0.0250  MAE: 0.8259 ± 0.0223  Prec@5: 0.5529 ± 0.0233  TopNAURC(k=100): 0.8118 ± 0.0102  runtime: 5.9044 ± 0.2935
# ------- SigmoidItemAsymFactoring
# RMSE: 1.0976 ± 0.0218  MAE: 0.9134 ± 0.0198  Prec@5: 0.5336 ± 0.0238  TopNAURC(k=100): 0.7864 ± 0.0178  runtime: 14.9367 ± 1.1114
# ------- BiasEstimator
# RMSE: 1.0210 ± 0.0251  MAE: 0.8218 ± 0.0214  Prec@5: 0.5535 ± 0.0321  TopNAURC(k=100): 0.8167 ± 0.0101  runtime: 9.4482 ± 1.6311
# ------- SVD
# RMSE: 1.0340 ± 0.0226  MAE: 0.8423 ± 0.0193  Prec@5: 0.5533 ± 0.0285  TopNAURC(k=100): 0.8214 ± 0.0093  runtime: 8.5963 ± 0.2888
