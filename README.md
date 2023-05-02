# Bankruptcy-Prediction-using-Pyspark-Python-and-AWS-
This project uses the Pyspark , Map reduce , Python to evaluate 5 machine learning models like logistic regression , Decision Tree, Random Forest , Multi Layer Perceptron(Neural Network) , Gradient Boosting Tree  
We evaluated the best model that will suit our model based upon the Model analysis , Evaluation Metrics , Parallelisim and Scalability , Hyperparameter tuning and Feature selection 
The data set had ~2.9 million data points . We did build the ML pipe line by perfroming EDA of the dataset and then implementing imputing , scaler assembling and smote analysis . The data was feeded into it by partition 
and caching . Further more to reduce the dimentinality reduction by PCA we again repartioned it . More over after every model was training we did perform unpersist
so as to free the memory and see good results of parallesim . 
