# Classification and Regression tasks on Used Car Analytics

## Task Description

The data consists of car listings from a used car business: they let their customers advertise their used cars in their website for a fee and connects those customers willing to sell their to millions of car dealers across the world.

The description of the dataset is given below.

The main aims of the project are the following:

1. A regression task to predict detail views column based on the information given in the other columns.

2. A classification task to predict the product tier of used cars from the information given in the other columns. The main challenge is classifying the used cars into three categories when the given training dataset contains highly imbalanced class labels (in the ratio of 1:3:136).

## Data Set

The original dataset is not provided due to confidentiality. A description of the columns in the dataset can be found below-

`Case_Study_Data.csv`- Original data used for training and evaluating the models (not given here due to reasons of confidentiality.)

`Data_Description.csv`- Description of the different columns in the dataset

## Results

The model evaluations were carried out on an independent test dataset that was not used during training or hyper-parameter optimisation using RandomizedSearch.

1. For the regression task of predicting the detail views column, Root Mean Squared Error (RMSE) was used as the evaluation metric. Out of all the different models tried, XGBoost provides the best score, more than one order of magnitude lower than the other models. 

The scores obtained from all the models tried are given below-

|Model     |   RMSE   | 
|---------|-----------------|
| XGBoost Regressor | 2.23 |
| Random Forest Regressor | 14.97 |
| Ridge Regressor | 41.61 |
| Lasso Regressor| 41.61 |
| Linear Regressor | 41.61 |

2. For the classification task of predicting the product tier of used cars, the training dataset had highly imbalanced multi-class class labels (3 classes). The averaged recall score (given by Sklearn’s balanced accuracy) was used as the evaluation metric. After trying out different algorithms and sampling methods, histogram-based gradient-boosting classifier within a BalancedBaggingClassifier provided the best averaged recall score, a big improvement over a naive baseline classifier (DummyClassifier) that predicts all classes with the same label.

The scores obtained from all the models tried are given below-

|Model     |   Averaged Recall   | 
|---------|-----------------------|
| Gradient-Boosting + BalancedBagging | 0.70 |
| XGBoost + Random Undersampling | 0.69 |
| Random Forest + Random Undersampling | 0.61 |
| Logistic Regression + Random Undersampling | 0.63 |
| Dummy Classifier | 0.33 |



#### Data Dictionary
```
article_id - unique article identifier

product_tier - premium status of the article

make_name - name of the car manufacturer

price - price of the article

first_zip_digit - first digit of the zip code of the region the article is offered in

first_registration_year - year of the first registration of the article

created_date - creation date of the listing

deleted_date - deletion date of the listing

search_views - number of times the article has been shown as a search result

detail_views - number of times the article has been clicked on

stock_days - Time in days between the creation of the listing and the deletion of the listing

ctr - Click through rate calculated as the quotient of detail_views over search_views

```

## Requirements
Python 3.7 or later and the following packages: 

`imbalanced_learn==0.8.0`

`matplotlib==3.4.`1`

`numpy==1.20.2`

`pandas==1.2.4`

`scikit-learn==0.24.2`

`scikit-lego==0.6.6`

`seaborn==0.11.1`


## Instructions for usage

There are three notebooks for the above mentioned tasks-

1. `01_cleaning_and_exporation.ipynb`: This notebook takes care of the exploratory data analysis, removal of missing values and outliers. It then exports the cleaned data into a new csv file called ‘cleaned_data.csv’ which is used in the subsequent classification and regression tasks.

2. `02_regression_detail_views_prediction.ipynb`: This notebook performs the regression task of predicting the ‘detail_views’ column (a continuous target) based on the other columns and new engineered features.

3. `03_classification_product_tier_prediction.ipynb`: This notebook performs the task of classifying the product tier of the used cars into three categories, which as highly imbalanced (in the ratio of 1:3:136). It investigates various approaches to deal with the class imbalance problem - by trying to balance class labels through random sampling, generating new synthetic data, using Random Forest and Gradient Boosted Trees in combination with the mentioned methods etc.
