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

# ------- Hybrid_odefault
# RMSE: 1.0170 ± 0.0256  MAE: 0.8178 ± 0.0223  Prec@5: 0.5599 ± 0.0218  TopNAURC(k=100): 0.8115 ± 0.0119  runtime: 59.9810 ± 1.3927
# ------- Hybrid_odefault_SVDpp
# RMSE: 1.0873 ± 0.0214  MAE: 0.9048 ± 0.0191  Prec@5: 0.5323 ± 0.0303  TopNAURC(k=100): 0.7926 ± 0.0143  runtime: 16.3069 ± 1.1609
# ------- Hybrid_odefault_AttributeBiasAdvanced
# RMSE: 1.0177 ± 0.0258  MAE: 0.8182 ± 0.0224  Prec@5: 0.5582 ± 0.0220  TopNAURC(k=100): 0.8096 ± 0.0123  runtime: 8.6887 ± 1.5159
# ------- Hybrid_ndefault
# RMSE: 1.0287 ± 0.0235  MAE: 0.8366 ± 0.0207  Prec@5: 0.5576 ± 0.0190  TopNAURC(k=100): 0.8337 ± 0.0095  runtime: 40.7413 ± 0.6421
# ------- Hybrid_ndefault_SigmoidUserAsymFactoring
# RMSE: 1.2101 ± 0.0171  MAE: 0.9903 ± 0.0177  Prec@5: 0.4795 ± 0.0318  TopNAURC(k=100): 0.5652 ± 0.0113  runtime: 11.8833 ± 0.5683
# ------- Hybrid_ndefault_AttributeBiasLight
# RMSE: 1.0286 ± 0.0228  MAE: 0.8362 ± 0.0197  Prec@5: 0.5552 ± 0.0236  TopNAURC(k=100): 0.8294 ± 0.0101  runtime: 3.8588 ± 0.1227
# ------- SigmoidItemAsymFactoring
# RMSE: 1.0977 ± 0.0193  MAE: 0.9134 ± 0.0176  Prec@5: 0.5283 ± 0.0269  TopNAURC(k=100): 0.7846 ± 0.0118  runtime: 15.5185 ± 0.7050
# ------- BiasEstimator
# RMSE: 1.0211 ± 0.0251  MAE: 0.8222 ± 0.0216  Prec@5: 0.5535 ± 0.0315  TopNAURC(k=100): 0.8160 ± 0.0103  runtime: 9.3960 ± 2.1370
# ------- SVD
# RMSE: 1.0337 ± 0.0228  MAE: 0.8421 ± 0.0190  Prec@5: 0.5533 ± 0.0261  TopNAURC(k=100): 0.8209 ± 0.0107  runtime: 8.7799 ± 0.4079
