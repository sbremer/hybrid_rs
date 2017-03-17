from pyspark.sql import SparkSession

from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
from pyspark.mllib.recommendation import ALS
import pyspark.sql.functions as spfun
from operator import add
from math import sqrt, floor
import numpy as np

import os
os.environ["SPARK_HOME"] = "/opt/apache-spark/"

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.executor.memory", "10g") \
    .config("spark.driver.memory", "2G") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

schema = StructType([StructField('uid', IntegerType(), True),
                     StructField('iid', IntegerType(), True),
                     StructField('rating', DoubleType(), True),
                     StructField('ts', IntegerType(), True)
                     ])

# Parameters
factors = 150
n_iter = 15
lmbd = 0.1

k = 5

# Load data
ratings = spark.read.csv('data/ml-100k/u.data', sep='\t', schema=schema)
ratings = ratings.drop('ts').withColumn('rnd', (spfun.floor(spfun.rand()*k)).cast(IntegerType()))

rmses = []

for a in range(k):
    train = ratings.filter(ratings.rnd == a).drop('rnd').rdd
    test = ratings.filter(ratings.rnd != a).drop('rnd').rdd

    n_train = train.count()
    n_test = test.count()

    model = ALS.train(train, factors, n_iter, lmbd)
    # model.predictAll(test.map(lambda x: (x[0], x[1])))
    predictions = model.predictAll(test.map(lambda x: (x[0], x[1])))

    predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]), x[2])) \
          .join(test.map(lambda x: ((x[0], x[1]), x[2]))) \
          .values()

    rmse = sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(n_test))
    rmses.append(rmse)

print('Crossval RMSE of MF-based CF: {}'.format(np.mean(rmses)))
