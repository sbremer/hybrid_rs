from evaluation.eval_script import evaluate_models_xval, print_results, EvalModel
from evaluation import evaluation
from hybrid_model.dataset import get_dataset

# Get dataset
dataset = get_dataset('ml100k')
# dataset = get_dataset('ml1m')

models = []

# Hybrid Model
from hybrid_model.hybrid import HybridModel
from hybrid_model.config import hybrid_config

model_type = HybridModel
config = hybrid_config
models.append(EvalModel(model_type.__name__, model_type, config))

evaluater = evaluation.Evaluation(evaluation.metrics_rmse_prec, evaluation.get_parting_all(10))

results = evaluate_models_xval(dataset, models, coldstart=False, evaluater=evaluater, n_fold=10, repeat=3)
print_results(results)

"""
------- HybridModel
=== Part full
rmse: 0.8915 ± 0.0062  prec@5: 0.8741 ± 0.0049

=== Part user_1
rmse: 0.8698 ± 0.0094  prec@5: 0.6756 ± 0.0214
=== Part user_2
rmse: 0.8683 ± 0.0107  prec@5: 0.7185 ± 0.0260
=== Part user_3
rmse: 0.8716 ± 0.0170  prec@5: 0.7824 ± 0.0129
=== Part user_4
rmse: 0.9156 ± 0.0218  prec@5: 0.8106 ± 0.0157
=== Part user_5
rmse: 0.9111 ± 0.0253  prec@5: 0.8997 ± 0.0118
=== Part user_6
rmse: 0.9028 ± 0.0282  prec@5: 0.9382 ± 0.0108
=== Part user_7
rmse: 0.9582 ± 0.0352  prec@5: 0.9692 ± 0.0108
=== Part user_8
rmse: 0.9521 ± 0.0427  prec@5: 0.9861 ± 0.0061
=== Part user_9
rmse: 0.9579 ± 0.0542  prec@5: 0.9946 ± 0.0046
=== Part user_10
rmse: 1.0080 ± 0.0347  prec@5: 0.9968 ± 0.0032

=== Part item_1
rmse: 0.8606 ± 0.0115  prec@5: 0.9508 ± 0.0033
=== Part item_2
rmse: 0.8855 ± 0.0142  prec@5: 0.9810 ± 0.0022
=== Part item_3
rmse: 0.9010 ± 0.0173  prec@5: 0.9901 ± 0.0022
=== Part item_4
rmse: 0.9165 ± 0.0200  prec@5: 0.9961 ± 0.0009
=== Part item_5
rmse: 0.9286 ± 0.0243  prec@5: 0.9977 ± 0.0013
=== Part item_6
rmse: 0.9898 ± 0.0306  prec@5: 0.9989 ± 0.0008
=== Part item_7
rmse: 0.9864 ± 0.0551  prec@5: 0.9991 ± 0.0014
=== Part item_8
rmse: 0.9875 ± 0.0732  prec@5: 0.9999 ± 0.0005
=== Part item_9
rmse: 1.0042 ± 0.0774  prec@5: 0.9994 ± 0.0015
=== Part item_10
rmse: 1.0718 ± 0.1142  prec@5: 0.9997 ± 0.0016

------- HybridModel_SVDpp
=== Part full
rmse: 0.8958 ± 0.0064  prec@5: 0.8728 ± 0.0048

=== Part user_1
rmse: 0.8732 ± 0.0098  prec@5: 0.6693 ± 0.0201
=== Part user_2
rmse: 0.8716 ± 0.0119  prec@5: 0.7154 ± 0.0262
=== Part user_3
rmse: 0.8765 ± 0.0165  prec@5: 0.7804 ± 0.0139
=== Part user_4
rmse: 0.9188 ± 0.0204  prec@5: 0.8103 ± 0.0146
=== Part user_5
rmse: 0.9174 ± 0.0245  prec@5: 0.8990 ± 0.0119
=== Part user_6
rmse: 0.9088 ± 0.0282  prec@5: 0.9383 ± 0.0107
=== Part user_7
rmse: 0.9616 ± 0.0352  prec@5: 0.9689 ± 0.0117
=== Part user_8
rmse: 0.9580 ± 0.0426  prec@5: 0.9862 ± 0.0064
=== Part user_9
rmse: 0.9630 ± 0.0511  prec@5: 0.9947 ± 0.0046
=== Part user_10
rmse: 1.0195 ± 0.0330  prec@5: 0.9967 ± 0.0033

=== Part item_1
rmse: 0.8641 ± 0.0120  prec@5: 0.9499 ± 0.0032
=== Part item_2
rmse: 0.8904 ± 0.0143  prec@5: 0.9807 ± 0.0024
=== Part item_3
rmse: 0.9056 ± 0.0169  prec@5: 0.9898 ± 0.0024
=== Part item_4
rmse: 0.9217 ± 0.0194  prec@5: 0.9959 ± 0.0011
=== Part item_5
rmse: 0.9354 ± 0.0240  prec@5: 0.9977 ± 0.0013
=== Part item_6
rmse: 0.9937 ± 0.0320  prec@5: 0.9989 ± 0.0009
=== Part item_7
rmse: 0.9913 ± 0.0540  prec@5: 0.9991 ± 0.0013
=== Part item_8
rmse: 0.9746 ± 0.0741  prec@5: 0.9998 ± 0.0007
=== Part item_9
rmse: 1.0238 ± 0.0767  prec@5: 0.9994 ± 0.0019
=== Part item_10
rmse: 1.0793 ± 0.1225  prec@5: 0.9995 ± 0.0020

------- HybridModel_AttributeBiasExperimental
=== Part full
rmse: 0.9236 ± 0.0061  prec@5: 0.8618 ± 0.0054

=== Part user_1
rmse: 0.9036 ± 0.0085  prec@5: 0.6371 ± 0.0206
=== Part user_2
rmse: 0.9005 ± 0.0102  prec@5: 0.6964 ± 0.0210
=== Part user_3
rmse: 0.9096 ± 0.0175  prec@5: 0.7561 ± 0.0195
=== Part user_4
rmse: 0.9519 ± 0.0238  prec@5: 0.7943 ± 0.0179
=== Part user_5
rmse: 0.9396 ± 0.0250  prec@5: 0.8921 ± 0.0112
=== Part user_6
rmse: 0.9228 ± 0.0305  prec@5: 0.9333 ± 0.0112
=== Part user_7
rmse: 0.9940 ± 0.0402  prec@5: 0.9659 ± 0.0107
=== Part user_8
rmse: 0.9764 ± 0.0463  prec@5: 0.9856 ± 0.0060
=== Part user_9
rmse: 0.9664 ± 0.0573  prec@5: 0.9945 ± 0.0051
=== Part user_10
rmse: 1.0291 ± 0.0344  prec@5: 0.9969 ± 0.0033

=== Part item_1
rmse: 0.9031 ± 0.0117  prec@5: 0.9438 ± 0.0039
=== Part item_2
rmse: 0.9207 ± 0.0138  prec@5: 0.9784 ± 0.0023
=== Part item_3
rmse: 0.9249 ± 0.0192  prec@5: 0.9892 ± 0.0027
=== Part item_4
rmse: 0.9327 ± 0.0222  prec@5: 0.9958 ± 0.0012
=== Part item_5
rmse: 0.9436 ± 0.0257  prec@5: 0.9978 ± 0.0013
=== Part item_6
rmse: 1.0107 ± 0.0279  prec@5: 0.9991 ± 0.0008
=== Part item_7
rmse: 1.0021 ± 0.0573  prec@5: 0.9993 ± 0.0015
=== Part item_8
rmse: 0.9971 ± 0.0769  prec@5: 0.9999 ± 0.0005
=== Part item_9
rmse: 1.0086 ± 0.0766  prec@5: 0.9994 ± 0.0015
=== Part item_10
rmse: 1.0748 ± 0.1134  prec@5: 0.9997 ± 0.0016
"""
