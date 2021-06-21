# Predicting the net hourly Electrical Energy Output of a Combined Cycle Power Plant
* Built regression models that can help predicting the net hourly electrical energy output of a Combined Cycle Power Plant
* The data was downloaded from the UCI Machine Learning Repository which is a website providing datasets for free.
* Applied 2 regression models (Multiple Linear Regression and Ridge Regression model) to find the best model.

## Programming Language & Packages
* Programming Language: Python
* Packages: numpy, pandas, sklearn

## Resources
* Dataset: https://archive.ics.uci.edu/ml/datasets/combined+cycle+power+plant

## Features/Variables
* Independent
  * Temperature (AT) in the range 1.81°C and 37.11°C
  * Exhaust Vacuum (V) in teh range 25.36-81.56 cm Hg
  * Ambient Pressure (AP) in the range 992.89-1033.30 milibar
  * Relative Humidity (RH) in the range 25.56% to 100.16%
* Dependent
  * Net hourly electrical energy output (PE) 420.26-495.76 MW

## Models Building
Firstly, I started by checking for NaN values and the outleirs, the dataset had no NaN values but by using the 95-5% quantiles, around 400 outliers were discovered. I don't have domin knowledge about hourly Electrical Energy Output of a Combined Cycle Power Plant, so, I chose to build the models before and after dealing with the outliers. Secondaly, I divding the dataset into train and test sets with train size of 80% and test size of 20%. I build 3 different models and evaluated them using Mean Abslout Error (MAE), and R2_score. I chose MAE because outliers aren’t particularly bad in for this type of model, and i used R2_score because getting model accuray in % is easy to understand, The models are:

* Multiple Linear Regression
* Ridge Regression - implemented Ridge model out of curiosity, I wanted to know if a normalized regression like Ridge would be effective for this problem.

## Models Performance
There is no difference in performace (models accuracy) between the 2 models before or after dealing with the outliers. after dealing with the outliers the model accuracy slightly increased.So, I believe best practice here will be just leaving the outliers as it is (since the model accuracy doesn't change by much). 
* Before dealing with the outliers:
  * Multiple Linear Regression & Ridge Regression - MAE : 3.585
  * Multiple Linear Regression & Ridge Regression - R2_score: 92.92%
* After dealing with the outliers:
  * Multiple Linear Regression & Ridge Regression - MAE : 3.343
  * Multiple Linear Regression & Ridge Regression - R2_score: 93.47%
* Multiple Linear Regression & Ridge Regression after dealing with the outliers - MAE
