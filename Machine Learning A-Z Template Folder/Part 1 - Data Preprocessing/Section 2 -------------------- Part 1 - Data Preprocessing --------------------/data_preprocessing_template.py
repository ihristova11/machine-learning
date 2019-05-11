# Data Preprocessing Templates

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
#we should specify the working directory
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

#taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder #use it for the dummy variables
labelencoder_X = LabelEncoder()
#we should not use this values since we cannot do equations with them
#because of their mathematical value
#so we need to use 'dummy variables' (represent them as an array I think)
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
#ecnode the dependent variable as 0 and 1 since it is a yes/no one
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y) 