#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import findspark
findspark.init()
import os
import pyspark
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.functions import col,isnan,when,count
from pyspark.sql.types import FloatType
from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler, PCA
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from sklearn.model_selection import train_test_split
from pyspark.sql import functions as F
import time


# In[ ]:


spark = SparkSession.builder.appName('bankruptcy_prediction').getOrCreate()


# In[ ]:


# Load the five years of data
num_partitions = 20
year1 = spark.read.csv("data/csv_result-1year.csv", header=True, inferSchema=True)
year2 = spark.read.csv("data/csv_result-2year.csv", header=True, inferSchema=True)
year3 = spark.read.csv("data/csv_result-3year.csv", header=True, inferSchema=True)
year4 = spark.read.csv("data/csv_result-4year.csv", header=True, inferSchema=True)
year5 = spark.read.csv("data/csv_result-5year.csv", header=True, inferSchema=True)
df_raw = year1.union(year2).union(year3).union(year4).union(year5)


# In[ ]:


# Filtering the data for 0s and ?

float_cols = ['Attr' + str(i) for i in range(1, 65)]

for col_name in float_cols:
    df_raw = df_raw.withColumn(col_name, when((col(col_name) == '0') | (col(col_name) == '?'), None).otherwise(col(col_name)))
    
# cast columns to float
for col_name in float_cols:
    df_raw = df_raw.withColumn(col_name, df_raw[col_name].cast(FloatType()))


# In[ ]:


df_raw.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_raw.columns]).show()


# In[ ]:


#Imputing missing values
imputer = Imputer(inputCols=df_raw.columns, outputCols=["{}_imputed".format(c) for c in df_raw.columns])
imputed_df = imputer.fit(df_raw).transform(df_raw)
imputed_df = imputed_df.drop(*imputed_df.columns[:66])
imputed_df = imputed_df.withColumnRenamed('class_imputed','label')


# In[ ]:


#Split train/test
train_df, test_df = imputed_df.randomSplit([.75, .25], seed=42)

train_df = train_df.repartition(num_partitions).cache()
test_df = test_df.repartition(num_partitions).cache()

X_train = train_df.drop('label', 'id_imputed')
Y_train = train_df.select('label')
X_test = test_df.drop('label', 'id_imputed')
Y_test = test_df.select('label')

# Assemble all feature columns into a single vector column
assembler = VectorAssembler(inputCols=X_train.columns, outputCol='features', handleInvalid="skip")
X_train = assembler.transform(X_train)
X_test = assembler.transform(X_test)

# Standardize the feature vector column
scaler = StandardScaler(inputCol='features', outputCol='scaled_features')
scaler_model = scaler.fit(X_train)
X_train = scaler_model.transform(X_train).drop('features')
X_test = scaler_model.transform(X_test).drop('features')

# Adding index to join Y_train and Y_test with X_train and X_test
X_train = X_train.withColumn('index', F.monotonically_increasing_id())
Y_train = Y_train.withColumn('index', F.monotonically_increasing_id())
X_test = X_test.withColumn('index', F.monotonically_increasing_id())
Y_test = Y_test.withColumn('index', F.monotonically_increasing_id())

X_train = X_train.join(Y_train, on='index', how='inner').drop('index')
X_test = X_test.join(Y_test, on='index', how='inner').drop('index')

X_train = X_train.repartition(num_partitions).cache()
X_test = X_test.repartition(num_partitions).cache()



# In[ ]:


# SMOTE implementation
from imblearn.over_sampling import SMOTE
from pyspark.ml.linalg import Vectors, DenseVector
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType

def dense_vector_to_list(vector: DenseVector):
    return vector.toArray().tolist()

dense_vector_to_list_udf = udf(dense_vector_to_list, ArrayType(DoubleType()))

X_train_pd = X_train.select(dense_vector_to_list_udf('scaled_features').alias('scaled_features_list'), 'label').toPandas()
X_train_list = X_train_pd['scaled_features_list'].tolist()
y_train_list = X_train_pd['label'].tolist()

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_list, y_train_list)

X_train_smote_df = spark.createDataFrame([(int(y), Vectors.dense(x)) for x, y in zip(X_train_smote, y_train_smote)], schema=['label', 'scaled_features'])


# In[ ]:


# PCA
start_time = time.time()
pca = PCA(k=10, inputCol='scaled_features', outputCol='reduced_features')
pca_model = pca.fit(X_train_smote_df)

pca_train = pca_model.transform(X_train_smote_df).select('label', 'reduced_features').withColumnRenamed('reduced_features', 'features')
pca_test = pca_model.transform(X_test).select('label', 'reduced_features').withColumnRenamed('reduced_features', 'features')

end_time = time.time()
print(f'PCA Train time: {end_time - start_time:.3f} seconds')


# In[ ]:


from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.evaluation import RegressionEvaluator

start_time = time.time()

lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=100)


paramGrid = (ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01, 0.001]).addGrid(lr.elasticNetParam, [0.0]).build())

crossval = CrossValidator(estimator=lr,estimatorParamMaps=paramGrid,evaluator=BinaryClassificationEvaluator(),numFolds=4)
# Train the Logistic Regression model                    
lr_model = crossval.fit(pca_train)                          
#Make predictions on the test set
predictions = lr_model.transform(pca_test)

# Evaluate the model
binary_evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
roc_auc = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderROC"})
multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label", metricName="rmse")
accuracy = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "accuracy"})
f1_score = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})
rmse = evaluator.evaluate(predictions)

end_time = time.time()
print(f'Logistic Regression evaluation time: {end_time - start_time:.3f} seconds')                         
print('Logistic Regression')
print("ROC-AUC: {:.5f}".format(roc_auc))
print("Accuracy: {:.6f}".format(accuracy))
print("F1 Score: {:.5f}".format(f1_score))
print("RMSE: {}".format(rmse))


# In[ ]:


# Calculate the elements of the confusion matrix
TN = predictions.filter('prediction = 0 AND label = prediction').count()
TP = predictions.filter('prediction = 1 AND label = prediction').count()
FN = predictions.filter('prediction = 0 AND label <> prediction').count()
FP = predictions.filter('prediction = 1 AND label <> prediction').count()
# show confusion matrix
predictions.groupBy('label', 'prediction').count().show()
# calculate metrics by the confusion matrix
accuracy = (TN + TP) / (TN + TP + FN + FP)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F =  2 * (precision*recall) / (precision + recall)
# calculate auc
evaluator = BinaryClassificationEvaluator()
auc = evaluator.evaluate(predictions, {evaluator.metricName: 'areaUnderROC'})
print('n precision: %0.3f' % precision)
print('n recall: %0.3f' % recall)
print('n accuracy: %0.3f' % accuracy)
print('n F1 score: %0.3f' % F)
print('AUC: %0.3f' % auc)


# In[ ]:


from pyspark.ml.classification import RandomForestClassifier

start_time = time.time()
# Train the RandomForestClassifier model
rf = RandomForestClassifier(featuresCol='features', labelCol='label', numTrees=100)

binary_evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

# Create a parameter grid for hyperparameter tuning
param_grid_rf = ParamGridBuilder()     .addGrid(rf.numTrees, [10, 20, 30])     .addGrid(rf.maxDepth, [5, 10, 15])     .build()

# Set up the cross-validator with the RandomForestClassifier, parameter grid, and desired number of folds
cross_validator_rf = CrossValidator(
    estimator=rf,
    estimatorParamMaps=param_grid_rf,
    evaluator=multi_evaluator,  # Use the MulticlassClassificationEvaluator from previous examples
    numFolds=4
)

rf_model = cross_validator_rf.fit(pca_train)

# Make predictions on the test set
rf_predictions = rf_model.transform(pca_test)



# Evaluate the model
rf_roc_auc = binary_evaluator.evaluate(rf_predictions, {binary_evaluator.metricName: "areaUnderROC"})
rf_accuracy = multi_evaluator.evaluate(rf_predictions, {multi_evaluator.metricName: "accuracy"})
rf_f1_score = multi_evaluator.evaluate(rf_predictions, {multi_evaluator.metricName: "f1"})
end_time = time.time()
print(f'Random Forest evaluation time: {end_time - start_time:.3f} seconds') 
print("Random Forest Classifier:")
print("ROC-AUC: {:.6f}".format(rf_roc_auc))
print("Accuracy: {:.6f}".format(rf_accuracy))
print("F1 Score: {:.6f}".format(rf_f1_score))


# In[ ]:


# Calculate the elements of the confusion matrix
TN = rf_predictions.filter('prediction = 0 AND label = prediction').count()
TP = rf_predictions.filter('prediction = 1 AND label = prediction').count()
FN = rf_predictions.filter('prediction = 0 AND label <> prediction').count()
FP = rf_predictions.filter('prediction = 1 AND label <> prediction').count()
# show confusion matrix
rf_predictions.groupBy('label', 'prediction').count().show()
# calculate metrics by the confusion matrix
accuracy = (TN + TP) / (TN + TP + FN + FP)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F =  2 * (precision*recall) / (precision + recall)
# calculate auc
evaluator = BinaryClassificationEvaluator()
auc = evaluator.evaluate(rf_predictions, {evaluator.metricName: 'areaUnderROC'})
print('n precision: %0.3f' % precision)
print('n recall: %0.3f' % recall)
print('n accuracy: %0.3f' % accuracy)
print('n F1 score: %0.3f' % F)
print('AUC: %0.3f' % auc)


# In[ ]:


from pyspark.ml.classification import MultilayerPerceptronClassifier

# Define the layers for the neural network:
# Input layer of size 10 (features), two hidden layers of size 20 and 10,
# and an output layer of size 2 (classes)

layers = [10, 20, 10, 2]
start_time = time.time()
# Create the trainer and set its parameters
mlp = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=42, featuresCol='features', labelCol='label')

binary_evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

param_grid = ParamGridBuilder()     .addGrid(mlp.stepSize, [0.01, 0.05, 0.1])     .addGrid(mlp.solver, ['l-bfgs', 'gd'])     .build()

# Set up the cross-validator with the Multilayer Perceptron Classifier, parameter grid, and desired number of folds
cross_validator = CrossValidator(
    estimator=mlp,
    estimatorParamMaps=param_grid,
    evaluator=multi_evaluator,  # Use the MulticlassClassificationEvaluator from previous examples
    numFolds=4
)


# Train the neural network model
mlp_model = mlp.fit(pca_train)

# Make predictions on the test set
mlp_predictions = mlp_model.transform(pca_test)

# Evaluate the model
mlp_roc_auc = binary_evaluator.evaluate(mlp_predictions, {binary_evaluator.metricName: "areaUnderROC"})
mlp_accuracy = multi_evaluator.evaluate(mlp_predictions, {multi_evaluator.metricName: "accuracy"})
mlp_f1_score = multi_evaluator.evaluate(mlp_predictions, {multi_evaluator.metricName: "f1"})
end_time = time.time()
print(f'Multilayer Perceptron evaluation time: {end_time - start_time:.3f} seconds') 
print("Multilayer Perceptron Classifier:")
print("ROC-AUC: {:.6f}".format(mlp_roc_auc))
print("Accuracy: {:.6f}".format(mlp_accuracy))
print("F1 Score: {:.6f}".format(mlp_f1_score))


# In[ ]:


# Calculate the elements of the confusion matrix
TN = mlp_predictions.filter('prediction = 0 AND label = prediction').count()
TP = mlp_predictions.filter('prediction = 1 AND label = prediction').count()
FN = mlp_predictions.filter('prediction = 0 AND label <> prediction').count()
FP = mlp_predictions.filter('prediction = 1 AND label <> prediction').count()
# show confusion matrix
mlp_predictions.groupBy('label', 'prediction').count().show()
# calculate metrics by the confusion matrix
accuracy = (TN + TP) / (TN + TP + FN + FP)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F =  2 * (precision*recall) / (precision + recall)
# calculate auc
evaluator = BinaryClassificationEvaluator()
auc = evaluator.evaluate(mlp_predictions, {evaluator.metricName: 'areaUnderROC'})
print('n precision: %0.3f' % precision)
print('n recall: %0.3f' % recall)
print('n accuracy: %0.3f' % accuracy)
print('n F1 score: %0.3f' % F)
print('AUC: %0.3f' % auc)


# In[ ]:


from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
start_time = time.time()
# Create a Decision Tree Classifier
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")

binary_evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
# Create a parameter grid for hyperparameter tuning
param_grid_dt = ParamGridBuilder()     .addGrid(dt.maxDepth, [5, 10, 15])     .addGrid(dt.maxBins, [16, 32, 64])     .build()

# Create a cross-validator
cross_validator_dt = CrossValidator(estimator=dt,
                                  estimatorParamMaps=param_grid_dt,
                                  evaluator=MulticlassClassificationEvaluator(metricName="f1"),
                                  numFolds=4)

# Train the Decision Tree Classifier using cross-validation
cv_model_dt = cross_validator_dt.fit(pca_train)

# Get the best Decision Tree Classifier
best_dt = cv_model_dt.bestModel

# Make predictions on the test set using the best Decision Tree Classifier
dt_predictions = best_dt.transform(pca_test)

# Evaluate the Decision Tree Classifier
binary_evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
# roc_auc = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderROC"})
# multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
dt_roc_auc = binary_evaluator.evaluate(dt_predictions, {binary_evaluator.metricName: "areaUnderROC"})
dt_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy").evaluate(dt_predictions)
dt_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1").evaluate(dt_predictions)
end_time = time.time()
print(f'Decision Tree Classifier evaluation time: {end_time - start_time:.3f} seconds') 
# Print the evaluation metrics
print("Decision Tree Classifier:")
print("ROC-AUC: {:.6f}".format(dt_roc_auc))
print("Accuracy: {:.6f}".format(dt_accuracy))
print("F1 Score: {:.6f}".format(dt_f1))


# In[ ]:


# Calculate the elements of the confusion matrix
TN = dt_predictions.filter('prediction = 0 AND label = prediction').count()
TP = dt_predictions.filter('prediction = 1 AND label = prediction').count()
FN = dt_predictions.filter('prediction = 0 AND label <> prediction').count()
FP = dt_predictions.filter('prediction = 1 AND label <> prediction').count()
# show confusion matrix
dt_predictions.groupBy('label', 'prediction').count().show()
# calculate metrics by the confusion matrix
accuracy = (TN + TP) / (TN + TP + FN + FP)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F =  2 * (precision*recall) / (precision + recall)
# calculate auc
evaluator = BinaryClassificationEvaluator()
auc = evaluator.evaluate(dt_predictions, {evaluator.metricName: 'areaUnderROC'})
print('n precision: %0.3f' % precision)
print('n recall: %0.3f' % recall)
print('n accuracy: %0.3f' % accuracy)
print('n F1 score: %0.3f' % F)
print('AUC: %0.3f' % auc)


# In[ ]:


from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
start_time = time.time()
# Define the GBT classifier with its parameters
gbt = GBTClassifier(maxDepth=5, maxIter=10)

binary_evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

# Define the parameter grid for hyperparameter tuning
param_grid = ParamGridBuilder()     .addGrid(gbt.maxDepth, [4, 8, 10])     .addGrid(gbt.maxIter, [2, 5, 10])     .build()

# Set up the cross-validator with the GBT classifier, parameter grid, and desired number of folds
cross_validator = CrossValidator(
    estimator=gbt,
    estimatorParamMaps=param_grid,
    evaluator=MulticlassClassificationEvaluator(metricName="f1"),
    numFolds=4
)

# Train the GBT model on the training set
gbt_model = cross_validator.fit(pca_train)

# Make predictions on the test set
gbt_predictions = gbt_model.transform(pca_test)

# Evaluate the GBT model using various evaluation metrics
gbt_accuracy = MulticlassClassificationEvaluator(metricName="accuracy").evaluate(gbt_predictions)
gbt_f1_score = MulticlassClassificationEvaluator(metricName="f1").evaluate(gbt_predictions)
gbt_roc_auc = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction").evaluate(gbt_predictions)

# Print the evaluation results
end_time = time.time()
print(f'GBT Classifier evaluation time: {end_time - start_time:.3f} seconds') 
print("GBT Classifier:")
print("Accuracy: {:.6f}".format(gbt_accuracy))
print("F1 Score: {:.6f}".format(gbt_f1_score))
print("ROC-AUC: {:.6f}".format(gbt_roc_auc))


# In[ ]:


# Calculate the elements of the confusion matrix
TN = gbt_predictions.filter('prediction = 0 AND label = prediction').count()
TP = gbt_predictions.filter('prediction = 1 AND label = prediction').count()
FN = gbt_predictions.filter('prediction = 0 AND label <> prediction').count()
FP = gbt_predictions.filter('prediction = 1 AND label <> prediction').count()
# show confusion matrix
gbt_predictions.groupBy('label', 'prediction').count().show()
# calculate metrics by the confusion matrix
accuracy = (TN + TP) / (TN + TP + FN + FP)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F =  2 * (precision*recall) / (precision + recall)
# calculate auc
evaluator = BinaryClassificationEvaluator()
auc = evaluator.evaluate(gbt_predictions, {evaluator.metricName: 'areaUnderROC'})
print('n precision: %0.3f' % precision)
print('n recall: %0.3f' % recall)
print('n accuracy: %0.3f' % accuracy)
print('n F1 score: %0.3f' % F)
print('AUC: %0.3f' % auc)

