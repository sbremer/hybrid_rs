import scripts
from scripts.basic_testing import get_results
from evaluation.eval_script import print_results

results = get_results('ml100k', coldstart=True, cs_type='user')
# results = get_results('ml1m', coldstart=True, cs_type='user')

print_results(results)

"""
ML100K:
------- HybridModel
=== Part full
rmse: 1.0169 ± 0.0256  prec@5: 0.5586 ± 0.0230

------- HybridModel_SVDpp
=== Part full
rmse: 1.0856 ± 0.0210  prec@5: 0.5317 ± 0.0281

------- HybridModel_AttributeBiasExperimental
=== Part full
rmse: 1.0173 ± 0.0254  prec@5: 0.5552 ± 0.0216

------- BiasEstimator
=== Part full
rmse: 1.0212 ± 0.0249  prec@5: 0.5559 ± 0.0300

------- SVD
=== Part full
rmse: 1.0343 ± 0.0225  prec@5: 0.5538 ± 0.0281

ML1M:
------- HybridModel
=== Part full
rmse: 0.9731 ± 0.0031  prec@5: 0.6220 ± 0.0065

------- HybridModel_SVDpp
=== Part full
rmse: 1.0078 ± 0.0060  prec@5: 0.5752 ± 0.0081

------- HybridModel_AttributeBiasExperimental
=== Part full
rmse: 0.9734 ± 0.0032  prec@5: 0.6221 ± 0.0068

------- BiasEstimator
=== Part full
rmse: 0.9853 ± 0.0030  prec@5: 0.6064 ± 0.0098

------- SVD
=== Part full
rmse: 0.9852 ± 0.0030  prec@5: 0.6070 ± 0.0107
"""
