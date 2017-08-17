import script_chdir
from scripts.test_final import get_results
from evaluation.eval_script import print_results

results = get_results('ml100k', coldstart=False)

print_results(results)

