# -*- coding: utf-8 -*-
"""
Created on 4/5/2024

SEHS4696 Machine Learning for Data Mining
Group Project
Topic: Trends in Discharges and Deaths


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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# ** 1. Load and Understand the Data
#
# Load the dataset into a dataframe called myDF
myDF = pd.read_csv('2.1M.csv', header=1)

#Check the dataset infomation (find out total 10 columns and 3090 rows)
print(myDF.head())
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

# -- 2.2 Handle missing data, if any
missing_data = myDF.isnull().sum()
print(missing_data)

# 2. Clean the Data

# Convert columns to appropriate data types
rent_columns = [col for col in myDF.columns if 'Grade' in col and 'Remarks' not in col]
for col in rent_columns:
    myDF[col] = pd.to_numeric(myDF[col], errors='coerce')

# Drop rows with missing values in the rent columns
myDF = myDF.dropna(subset=rent_columns)

# Create the target variable (rental grade)
myDF['Target'] = np.argmax(myDF[rent_columns].values, axis=1)

# Convert 'District' to categorical and create dummy variables
mymyDF = pd.get_dummies(myDF, columns=['District'], drop_first=True)

# Check if 'District' column is present after one-hot encoding
if 'District' in myDF.columns:
    features = myDF.columns.tolist()
    features.remove('Target')  # Remove the target variable from features
else:
    print("Error: 'District' column not found after one-hot encoding.")
    exit()

# 3. Split the Data into Training and Testing Sets
X = myDF[features]
y = myDF['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train a Logistic Regression Model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# 5. Evaluate the Model
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Classification Report
print('\nClassification Report:\n', classification_report(y_test, y_pred))

# Confusion Matrix
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
