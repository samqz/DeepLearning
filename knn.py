# Import all necessary libraries 

from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
import numpy as np
import math
from pyspark.mllib.evaluation import MulticlassMetrics
import time


if __name__ == "__main__":

    start =time.time()

    #initinalization
    spark = SparkSession \
        .builder \
        .appName("knn") \
        .getOrCreate()

    #Load the data in csv file
    train_datafile = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv"
    test_datafile= "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv"
    num_train_samples = 60000
    num_test_samples = 10000

    train_df = spark.read.csv(train_datafile,header=False,inferSchema="true")
    test_df = spark.read.csv(test_datafile,header=False,inferSchema="true")

    #Converting to Vector row
    assembler_train_features = VectorAssembler(inputCols=train_df.columns[1:],outputCol="features")
    train_vectors = assembler_train_features.transform(train_df).select("_c0","features")

    assembler_test_features = VectorAssembler(inputCols=test_df.columns[1:],outputCol="features")
    test_vectors = assembler_test_features.transform(test_df).select("_c0","features")

    #PCA 
    pca_value = 50
    pca = PCA(k=pca_value, inputCol="features", outputCol="pca") 
    train_model = pca.fit(train_vectors) 
    pca_train_result = train_model.transform(train_vectors).select('pca')
    pca_test_result = train_model.transform(test_vectors).select('_c0','pca')

    #Inspect the PCA and reshape it to 2d array
    local_pca_train = np.array(pca_train_result.collect()).reshape(num_train_samples,pca_value)
    #local_pca_test = np.array(pca_test_result.collect()).reshape(num_test_samples,pca_value)


    #set the labels to array
    labels = np.array(train_vectors.select("_c0").rdd.collect())
    labels = labels.reshape(num_train_samples)

    train_array = local_pca_train

    test_rdd = pca_test_result.rdd

    #define knn function
    def KNN(data):
        k = 5
        distance = (((train_array-np.array(data[1]))**2).sum(axis=1))**0.5
        sort_indeces = distance.argsort()
        labelCount = dict()
        for j in range(k):
            label = labels[sort_indeces[j]]
            labelCount[label] = labelCount.get(label,0) + 1

        num = 0  
        for key, value in labelCount.items():  
            if value > num:  
                num = value  
                pre_label = key  
        return (float(pre_label), float(data[0]))

    #Precision, Recall and f1-score
    predictionAndLabels = test_rdd.map(KNN)
    metrics = MulticlassMetrics(predictionAndLabels)

    print('label   precision    recall    f1-score')
    for i in range(10):
        print('  {}     {:.5f}     {:.5f}     {:.5f}'.format(i, metrics.precision(i), metrics.recall(i), metrics.fMeasure(float(i), 1.0)))
          
    end = time.time()
    print()
    print('Execution time: {} Seconds'.format(end-start))
    

    
