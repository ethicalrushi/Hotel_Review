#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 21:20:24 2018

@author: rushikesh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('chennai_reviews.csv')

#Removing unnecessary features
dataset = dataset.drop(['Unnamed: 8', 'Unnamed: 6', 'Unnamed: 7','Unnamed: 5'], axis=1)

#Removing sentiments having text that is len more than or equal to 2
df = dataset[[(len(str(x))<2) for x in dataset['Sentiment']]]


X = df.iloc[:,[2]]
y = df.iloc[:, 3]

#Splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Buildign a word count vector 
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()


X_train_vectorized = vect.fit_transform(X_train['Review_Text'])
a = X_train_vectorized.toarray()


#Applying Logistic Regreesion
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)


from sklearn.metrics import roc_auc_score

predictions = model.predict(vect.transform(X_test['Review_Text']))

print('AUC: ', roc_auc_score(y_test, predictions))


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)
#got 0.80 accuracy evaluated using confusion matrix




