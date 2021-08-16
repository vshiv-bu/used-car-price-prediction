# Used Car Price Prediction based on Craigslist Data
## What is it?
An Interactive Python Jupyter Notebook (ipynb) that uses this Craigslist used car dataset to predict used car prices based on last 20 years' worth of data

These algorithms have been used to best predict the value of used cars, based on 10 features
*	Linear Regression
*	Decision Trees 
*	Bagging 
*	Random Forest
*	Adaptive Boosting 
*	Gradient Boosting 
*	XGBoost

## How does it work?
* The notebook starts by reading data into a pandas data frame
* It then uses the module missingno to understand the completeness of the dataset and removes the features which are mostly incomplete. Then through some initial data cleaning, it drops all duplicate records and NaNs

*	Moving on to exploratory data analysis, the notebook uses histograms, category, scatter & box plots to identify relationships and perform second level data cleaning – some examples include filtering data between manufacturing years 2000 and 2020, removing records with odometer readings <100 and >200k miles etc.

*	On this cleaned dataset, categorical features are Label Encoded and correlation between features is analyzed 

*	Finally, the data is split into training and testing subsets (70-30) and aforementioned algorithms are applied in the same order. As an additional step, GridSearchCV is used to run 48 iterations on number of estimators, learning rates and maximum depth of trees to get the best accuracy for XGBoost

## Where can the notebook be found?

https://www.kaggle.com/vinodshiv/used-car-price-prediction-20-years-data

