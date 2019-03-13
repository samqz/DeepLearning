# Import all necessary libraries

from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
import numpy as np
from pyspark.ml.classification import LogisticRegression


if __name__ == "__main__":

    spark = SparkSession \
        .builder \
	.appName("logistic regression") \
	.getOrCreate()

    train_datafile = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv"
    test_datafile= "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv"
    num_train_samples = 60000
    num_test_samples = 10000

    train_df = spark.read.csv(train_datafile,header=False,inferSchema="true")
    test_df = spark.read.csv(test_datafile,header=False,inferSchema="true")

    assembler_train_features = VectorAssembler(inputCols=train_df.columns[1:],outputCol="features")
    train_vectors = assembler_train_features.transform(train_df).select("_c0","features")

    assembler_test_features = VectorAssembler(inputCols=test_df.columns[1:],outputCol="features")
    test_vectors = assembler_test_features.transform(test_df).select("_c0","features")

    train_vectors = train_vectors.selectExpr("_c0 as label", "features as features")
    test_vectors = test_vectors.selectExpr("_c0 as label", "features as features")

    #get dataframe after pca
    pca_value = 60
    pca = PCA(k=pca_value, inputCol="features", outputCol="pca")
    train_model = pca.fit(train_vectors)
    pca_train_result = train_model.transform(train_vectors).select('label','pca')
    pca_test_result = train_model.transform(test_vectors).select('label','pca')
    pca_train_result = pca_train_result.selectExpr("label as label", "pca as features")
    pca_test_result = pca_test_result.selectExpr("label as label", "pca as features")
    # pca_train_result.show(10)

    # Fit the model
    lr = LogisticRegression()
    lrModel = lr.fit(pca_train_result)

    prediction = lrModel.transform(pca_test_result)
    result = prediction.select("prediction" , "label")

    result2 = np.array(result.rdd.collect())
    count = 0
    for i in range(len(result2)):
        if result2[i][0] == result2[i][1]:
            count += 1
    accuracy = (count/10000)
    print('{}%'.format(accuracy*100))

