from surprise import BaselineOnly, NMF, SVD
from surprise import Dataset
from surprise import evaluate, print_perf


# Load the movielens-100k dataset (download it if needed),
# and split it into 3 folds for cross-validation.
data = Dataset.load_builtin('ml-100k')
data.split(n_folds=3)

# # We'll use the famous SVD algorithm.
# algo = SVD()
#
# # Evaluate performances of our algorithm on the dataset.
# perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
#
# print_perf(perf)

# We'll use the famous SVD algorithm.
algos = [BaselineOnly(), NMF(n_factors=100), SVD(n_factors=100)]

for algo in algos:
    # Evaluate performances of our algorithm on the dataset.
    perf = evaluate(algo, data, measures=['RMSE', 'MAE'], verbose=0)
    print_perf(perf)
