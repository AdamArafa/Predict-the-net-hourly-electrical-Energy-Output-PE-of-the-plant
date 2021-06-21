# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 15:16:31 2021

@author: arafa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae



# read the data to Pandas dataframe

# the data columns are Features consist of hourly average ambient variables:
# 1. Temperature (AT) in the range 1.81°C and 37.11°C
# 2. Exhaust Vacuum (V) in teh range 25.36-81.56 cm Hg
# 3. Ambient Pressure (AP) in the range 992.89-1033.30 milibar
# 4. Relative Humidity (RH) in the range 25.56% to 100.16%
# 5. Net hourly electrical energy output (PE) 420.26-495.76 MW

df = pd.read_excel('data/d_new.xlsx')

df.head()

# check for NaN values
df.isna().sum()

#check for outleirs
# running the function below shows that around 10% of our data is outleirs, quantile (%) can be changed but 
# 95%-5% are the most common to use. 

def check_outlier(input):
        max_thresold = df[input].quantile(0.95)  
        min_thresold = df[input].quantile(0.05)
        print("========== For {}=========".format(input))
        print("max_thresold is {} and min_thresold is {}". format(max_thresold, min_thresold))
        print("max_value is {} and min_value is {}". format(np.max(df[input]), np.min(df[input])))
        df_temp = df[(df[input] < min_thresold) | (df[input] > max_thresold)]
        print("Number of Outliers Detected in {}:".format(input), df_temp.shape[0])
        print("------------------------------------------------------------")
        
        
for c in df.columns:
    check_outlier(c)
    
# I will run multiple linear regression model and Ridge model on the dataset before and after dealing with the outleirs and measure the performance of 
# each model

# X is all the columns but 'PE', y is 'PE'
X = df.drop('PE', axis = 1).values
y = df['PE'].values

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Build multiple linear regression model
lrm_model = LinearRegression()

# running cross validation and calculate the score of the model
mse = cross_val_score(lrm_model, X_train, y_train, scoring='neg_mean_squared_error',cv=5)
mean_mse = np.mean(mse)
mean_mse # -20.888

lrm_model.fit(X_train, y_train)

# Build Ridge model usign the 

parameters = {'alpha':[(i/100) for i in range(100)]} # [0.1, 0.2, .......,0.99]


r_model = Ridge()

# define the grid search
Ridge_regressor= GridSearchCV(r_model, parameters, scoring='neg_mean_squared_error',cv=5)
Ridge_regressor.fit(X_train, y_train)

Ridge_regressor.best_score_ # best score is 20.888. There is no differnec between Ridge model and multiple linear regression model
Ridge_regressor.best_params_ # best alpha is 1


# make predictions for linear regression model
lrm_predictions = lrm_model.predict(X_test)

score = r2_score(y_test, lrm_predictions)

#the model has an accuracy score of 92.92%
print('The multiple linear regression model has an accuracy of {}%'.format(round(score * 100, 2))) 


# make prediction for Ridge regression
r_predictions = Ridge_regressor.predict(X_test)

score = r2_score(y_test, r_predictions)

#the model has an accuracy score of 92.92%
print('The multiple Ridge model has an accuracy of {}%'.format(round(score * 100, 2))) 


# evaluate the models by comparing predictions with actual values
mae(y_test,lrm_predictions)
mae(y_test,r_predictions)

# multiple linear regression - test on a single value 
pred_single_value = lrm_model.predict(X_train[0].reshape(1,-1))[0]
actual_value = y_train[0]

print('the predicted value is {} and the actual value is {}'.format(pred_single_value, actual_value))


# dealing with the outleirs and train the model again
# replace all the values that are higher than the 
# the code below is exactly the same as above, the only different is the data itself, the outliers are replaced 
# with other valus 

def fix_outlier(input):
    for i in input:
        max_thresold = df[input].quantile(0.95)  
        min_thresold = df[input].quantile(0.05)
        df[input] = np.where(df[input] > max_thresold,max_thresold,df[input])
        df[input] = np.where(df[input] < min_thresold,min_thresold,df[input])
    
for c in df.columns:
    fix_outlier(c)
    
    
# X is all the columns but 'PE', y is 'PE'
X = df.drop('PE', axis = 1).values
y = df['PE'].values

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Build multiple linear regression model
lrm_model = LinearRegression()


mse = cross_val_score(lrm_model, X_train, y_train, scoring='neg_mean_squared_error',cv=5)
mean_mse = np.mean(mse)
mean_mse # -20.888

lrm_model.fit(X_train, y_train)

# Build Ridge model usign the 

parameters = {'alpha':[(i/100) for i in range(100)]}
r_model = Ridge()

# define the grid search
Ridge_regressor= GridSearchCV(r_model, parameters, scoring='neg_mean_squared_error',cv=5)
Ridge_regressor.fit(X_train, y_train)

Ridge_regressor.best_score_ # best score is 20.888. There is no differnec between Ridge model and multiple linear regression model
Ridge_regressor.best_params_ # best alpha is 1


# make predictions for linear regression model
lrm_predictions = lrm_model.predict(X_test)

score = r2_score(y_test, lrm_predictions)

#the model has an accuracy score of 93.47%
print('The multiple linear regression model has an accuracy of {}%'.format(round(score * 100, 2))) 


# make prediction for Ridge regression
r_predictions = Ridge_regressor.predict(X_test)

score = r2_score(y_test, r_predictions)

#the model has an accuracy score of 93.47%
print('The multiple Ridge model has an accuracy of {}%'.format(round(score * 100, 2))) 

# evaluate the models by comparing predictions with actual values
mae(y_test,lrm_predictions)
mae(y_test,r_predictions)
