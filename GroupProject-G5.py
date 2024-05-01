# -*- coding: utf-8 -*-
"""
Created on 4/5/2024

SEHS4696 Machine Learning for Data Mining
Group Project
Topic: Private Offices


@author: SEHS4696 MACHINE LEARNING FOR DATA MINING (Group 5)
@Student names and numbers: 22060542S CHAN Ka Lok
                            22059790S CHEN Yuanhang
                            22059727S FUNG Ho Wai
                            22059069S HO Man Kit
                            22053502S LUI Wing Ho





"""

#import the necessary package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score


# ** 1. Load and Understand the Data
#
# Load the dataset into a dataframe called myDF
myDF = pd.read_csv('2.1M.csv', header=1)
myDF= myDF.set_index('Month')
#df.index = pd.to_datetime(df.index)
myDF.index = pd.to_datetime(myDF.index, format="%b-%y")
#Check the dataset infomation
print(myDF.info())


# ** 2. Data Preprocessing
#
# -- 2.1 Drop Unnecessary Features
myDF.drop('Grade A Sheung Wan - Remarks',axis=1,inplace=True)
myDF.drop('Grade A Central - Remarks',axis=1,inplace=True)
myDF.drop('Grade A Wan Chai / Causeway Bay - Remarks',axis=1,inplace=True)
myDF.drop('Grade A North Point / Quarry Bay - Remarks',axis=1,inplace=True)
myDF.drop('Grade A Tsim Sha Tsui - Remarks',axis=1,inplace=True)
myDF.drop('Grade A Yau Ma Tei / Mong Kok - Remarks',axis=1,inplace=True)
myDF.drop('Grade A Kowloon Bay / Kwun Tong - Remarks',axis=1,inplace=True)

myDF.drop('Grade B Sheung Wan - Remarks',axis=1,inplace=True)
myDF.drop('Grade B Central - Remarks',axis=1,inplace=True)
myDF.drop('Grade B Wan Chai / Causeway Bay - Remarks',axis=1,inplace=True)
myDF.drop('Grade B North Point / Quarry Bay - Remarks',axis=1,inplace=True)
myDF.drop('Grade B Tsim Sha Tsui - Remarks',axis=1,inplace=True)
myDF.drop('Grade B Yau Ma Tei / Mong Kok - Remarks',axis=1,inplace=True)
myDF.drop('Grade B Kowloon Bay / Kwun Tong - Remarks',axis=1,inplace=True)

myDF.drop('Grade C Sheung Wan - Remarks',axis=1,inplace=True)
myDF.drop('Grade C Central - Remarks',axis=1,inplace=True)
myDF.drop('Grade C Wan Chai / Causeway Bay - Remarks',axis=1,inplace=True)
myDF.drop('Grade C North Point / Quarry Bay - Remarks',axis=1,inplace=True)
myDF.drop('Grade C Tsim Sha Tsui - Remarks',axis=1,inplace=True)
myDF.drop('Grade C Yau Ma Tei / Mong Kok - Remarks',axis=1,inplace=True)
myDF.drop('Grade C Kowloon Bay / Kwun Tong - Remarks',axis=1,inplace=True)

# -- 2.2 Encode strings into numbers
myDF['Grade A Kowloon Bay / Kwun Tong'] = myDF['Grade A Kowloon Bay / Kwun Tong'].replace(np.nan, 0)
myDF['Grade B Kowloon Bay / Kwun Tong'] = myDF['Grade B Kowloon Bay / Kwun Tong'].replace(np.nan, 0)
myDF['Grade C Kowloon Bay / Kwun Tong'] = myDF['Grade C Kowloon Bay / Kwun Tong'].replace(np.nan, 0)

replacement_dict = {"-": None}
myDF.replace(replacement_dict, inplace=True)

# -- 2.3 Handle missing data, if any
missing_data = myDF.isnull().sum()
print(missing_data)

if missing_data.any():
    columns_to_impute = ['Grade A Sheung Wan', 'Grade A Central', 'Grade A Wan Chai / Causeway Bay', 'Grade A North Point / Quarry Bay', 'Grade A Tsim Sha Tsui', 'Grade A Yau Ma Tei / Mong Kok', 'Grade A Kowloon Bay / Kwun Tong','Grade B Sheung Wan', 'Grade B Central', 'Grade B Wan Chai / Causeway Bay', 'Grade B North Point / Quarry Bay', 'Grade B Tsim Sha Tsui', 'Grade B Yau Ma Tei / Mong Kok', 'Grade B Kowloon Bay / Kwun Tong', 'Grade C Sheung Wan', 'Grade C Central', 'Grade C Wan Chai / Causeway Bay', 'Grade C North Point / Quarry Bay', 'Grade C Tsim Sha Tsui', 'Grade C Yau Ma Tei / Mong Kok', 'Grade C Kowloon Bay / Kwun Tong']
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    myDF[columns_to_impute] = pd.DataFrame(imputer.fit_transform(myDF[columns_to_impute]), columns=columns_to_impute)
    
# -- 2.4 Feature Selection
plt.plot(myDF['Month'], myDF['Grade A Sheung Wan'], 'x')
plt.grid(True)

plt.plot(myDF['Month'], myDF['Grade A Central'], 'o')
plt.plot(myDF['Month'], myDF['Grade B Central'], 'o')
plt.plot(myDF['Month'], myDF['Grade C Central'], 'o')

#df['Grade A Central'] = df.target

X = myDF.index.month
y = myDF['Grade A Central']
X = np.array(X).reshape(-1, 1)
x_train, x_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state = 42 )
print(x_train)
print(Y_train)


model = LinearRegression()
model.fit(x_train,Y_train)

# Predicting rent for the test set
Y_pred = model.predict(x_test)

# Evaluating the model
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Optionally, you can print the coefficients and intercept of the model
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)