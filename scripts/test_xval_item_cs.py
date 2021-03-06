import script_chdir
from scripts.basic_testing import get_results
from evaluation.eval_script import print_results

results = get_results('ml100k', coldstart=True, cs_type='item')
# results = get_results('ml1m', coldstart=True, cs_type='item')

print_results(results)

"""
ML100K:
------- HybridModel
=== Part full
rmse: 1.0563 ± 0.0148  prec@5: 0.6622 ± 0.0209

------- HybridModel_SVDpp
=== Part full
rmse: 1.1335 ± 0.0132  prec@5: 0.6035 ± 0.0240

------- HybridModel_AttributeBiasExperimental
=== Part full
rmse: 1.0517 ± 0.0126  prec@5: 0.6610 ± 0.0219

------- BiasEstimator
=== Part full
rmse: 1.0742 ± 0.0124  prec@5: 0.6035 ± 0.0240

------- SVD
=== Part full
rmse: 1.0680 ± 0.0118  prec@5: 0.6035 ± 0.0240

ML1M:
------- HybridModel
=== Part full
rmse: 1.0673 ± 0.0164  prec@5: 0.6063 ± 0.0159

------- HybridModel_SVDpp
=== Part full
rmse: 1.1150 ± 0.0175  prec@5: 0.5413 ± 0.0214

------- HybridModel_AttributeBiasExperimental
=== Part full
rmse: 1.0666 ± 0.0162  prec@5: 0.6063 ± 0.0164

------- BiasEstimator
=== Part full
rmse: 1.0748 ± 0.0155  prec@5: 0.5413 ± 0.0214

------- SVD
=== Part full
rmse: 1.0750 ± 0.0150  prec@5: 0.5413 ± 0.0214
"""
