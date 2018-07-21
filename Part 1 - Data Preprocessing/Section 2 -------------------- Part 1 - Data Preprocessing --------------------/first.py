# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1]
y=dataset.iloc[:,3]
"""
#Filling Missing Data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x.iloc[:,1:3]) 
x.iloc[:,1:3]=imputer.transform(x.iloc[:,1:3])

#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder
labelEncoder_x=LabelEncoder()
x.iloc[:,0]=labelEncoder_x.fit_transform(x.iloc[:,0])

labelEncoder_y=LabelEncoder()
y=labelEncoder_y.fit_transform(y)

the above code will give every country name a special code that is a
special number like here 0, 1, 2...butt this is sometimes tough to implement
as we have to do calculations so here we instead do some thing like
below

from sklearn.preprocessing import OneHotEncoder	
onehotencoder = OneHotEncoder(categorical_features =[0])
x=onehotencoder.fit_transform(x).toarray()


from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

"""
