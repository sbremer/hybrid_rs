import scripts
from scripts.basic_testing import get_results
from evaluation.eval_script import print_results

results = get_results('ml100k', coldstart=False)

print_results(results)

"""
------- HybridModel
=== Part full
rmse: 0.8962 ± 0.0027  prec@5: 0.7809 ± 0.0030

------- HybridModel_SVDpp
=== Part full
rmse: 0.8996 ± 0.0027  prec@5: 0.7786 ± 0.0030

------- HybridModel_AttributeBiasExperimental
=== Part full
rmse: 0.9265 ± 0.0044  prec@5: 0.7584 ± 0.0022

------- BiasEstimator
=== Part full
rmse: 0.9417 ± 0.0049  prec@5: 0.7483 ± 0.0024

------- SVD
=== Part full
rmse: 0.9246 ± 0.0044  prec@5: 0.7646 ± 0.0028
"""
