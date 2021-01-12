### Forecasting Sales of Walmart in given Data

<p align="center"><img src"https://github.com/Starkultra/Walmart-Sales-Prediction/blob/master/assets/walmart-store.jpg" width=30%><p>
 
Problem Statement

Demand Forecast is one of the key tasks in Supply Chain and Retail Domain in general. It is key in effective operation and optimization of retail supply chain. 
Effectively solving this problem requires knowledge about a wide range of tricks in Data Sciences and good understanding of ensemble techniques. 
You are required to predict sales for each Store-Day level for one month. 
All the features will be provided and actual sales that happened during that month will also be provided for model evaluation. 
Dataset Snapshot

**Training Data Description:** Historic sales at Store-Day level for about two years for a retail giant, for more than 1000 stores. 
Also, other sale influencers like, whether on a particular day the store was fully open or closed for renovation, holiday and special event details, are also provided. 

 

Exploratory Data Analysis (EDA) and Linear Regression:

1.      Transformed the variables by using data manipulation techniques like, One-Hot Encoding 
2.      Performed an EDA (Exploratory Data Analysis) to see the impact of variables over Sales.
3.      Apply Linear Regression to predict the forecast and evaluate different accuracy metrices like RMSE (Root Mean Squared Error)
         and MAE(Mean Absolute Error) and determine which metric makes more sense. Can there be a better accuracy metric?
         a)      Train a single model for all stores, using storeId as a feature.
         b)      Train separate model for each store.
         c)      Observation from previous 3 models **Second one performs better than first one [In the first case, parameters are shared and not very free but not in second case]**
         d)      Try Ensemble of b) and c). **No improvement**
         e)      Use Regularized Regression. It should perform better in an unseen test set. **No Improvement**
         f)      Open-ended modeling to get possible predictions.


Other Regression Techniques:

1. When store is closed, sales = 0. used this insight be used for Data Cleaning Perform this and retrain the model.
2. Use Non-Linear Regressors like Random Forest or other Tree-based Regressors(XGBoostRegressor).
       a)    Trained a single model for all stores, where storeId can be a feature.
       b)    Trained separate models for each store.
       Note: Dimensional Reduction techniques like, PCA and Tree’s Hyperparameter Tuning will be required. Cross-validate to find the
                  best parameters. Infer the performance of both the models. 
3 Compare the performance of Linear Model and Non-Linear Model from the previous observations. Which performs better and why?
4. Train a Time-series model on the data taking time as the only feature. This will be a store-level training.
       a)    Identify yearly trends and seasonal months
 


Implementing Neural Networks:

Train a LSTM on the same set of features and compare the result with traditional time-series model.
Cluster stores using sales and customer visits as features. Find out how many clusters or groups are possible. Also visualize the results.
Is it possible to have separate prediction models for each cluster? Compare results with the previous models.


Applying ANN:

1.     Use ANN (Artificial Neural Network) to predict Store Sales.
       a)    Fine-tune number of layers,
       b)    Number of Neurons in each layers .
       c)    Experiment in batch-size.
       d)    Experiment with number of epochs. Carefully observe the loss and accuracy? What are the observations?
       e)    Play with different  Learning Rate  variants of Gradient Descent like Adam, SGD, RMS-prop.
       f)    Which activation performs best for this use case and why?
       g)    Check how it performed in the dataset, calculate RMSE.
2.    Use Dropout for ANN and find the optimum number of clusters (clusters formed considering the features: sales and customer
       visits). Compare model performance with traditional ML based prediction models. 
3.    Find the best setting of neural net that minimizes the loss and can predict the sales best. Use techniques like Grid
       search, cross-validation and Random search.
