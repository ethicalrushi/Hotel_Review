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

X_pre = df.iloc[:,[0,2]]
X = df.iloc[:,[2]]
y = df.iloc[:, 3]

#Splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_pre_train, X_pre_test, y_pre_test, y_pre_test = train_test_split(X_pre, y, test_size = 0.2, random_state = 0)
#########---------------------------------##########################
########################Model-1#########################################

#Buildign a word count vector 
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()


X_train_vectorized = vect.fit_transform(X_train['Review_Text'])




#Applying Logistic Regreesion
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)


#Applying Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=5000)
classifier.fit(X_train_vectorized, y_train)







predictions = model.predict(vect.transform(X_test['Review_Text']))
predictions1 = classifier.predict(vect.transform(X_test['Review_Text']))




from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)
cm1 = confusion_matrix(y_test, predictions1)
#got 0.80with RandomForest

#got 0.80 accuracy evaluated using confusion matrix with logistic regrssion


feature_names = np.array(vect.get_feature_names())
sorted_coef_index = model.coef_[0].argsort()
#printing most negative and positive words
print('Smallest Coefs: \n{} \n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{} \n'.format(feature_names[sorted_coef_index[:-11:-1]]))


###########------------------------###############################
#######################Model-2##########################################

#Tf-idf term weighting 
from sklearn.feature_extraction.text import TfidfVectorizer

#min_df = 5 will remove words from vocabulary that appears in less than five doc
vect = TfidfVectorizer(min_df=5).fit(X_train['Review_Text'])
len(vect.get_feature_names())
#reduced features to 1808 from 5619

X_train_vectorized = vect.transform(X_train['Review_Text'])

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
predictions = model.predict(vect.transform(X_test['Review_Text']))
cm = confusion_matrix(y_test, predictions)
#Though we reduced the features from 5619 to 1808 the accuracy reduced to 0.78 from 0.80


feature_names = np.array(vect.get_feature_names())

sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()
print('Smallest Coefs: \n{} \n'.format(feature_names[sorted_tfidf_index[:10]]))
print('Largest Coefs: \n{} \n'.format(feature_names[sorted_tfidf_index[:-11:-1]]))


##############-----------------------------#########################
#########################Model-3#################################


#using n grams to consider two words together like not good
#Since it is very different from good

vect = CountVectorizer(min_df = 5, ngram_range = (1,3)).fit(X_train['Review_Text'])
X_train_vectorized = vect.transform(X_train['Review_Text'])
len(vect.get_feature_names())
a = X_train_vectorized.toarray()
#since we considered the ngrams the features increased to 6137

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
predictions = model.predict(vect.transform(X_test['Review_Text']))

cm = confusion_matrix(y_test, predictions)
#this increased our accuracy to 0.82






feature_names = np.array(vect.get_feature_names())
sorted_coef_index = model.coef_[0].argsort()


#printing most negative and positive words
print('Smallest Coefs: \n{} \n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{} \n'.format(feature_names[sorted_coef_index[:-11:-1]]))





#Naive Bayes approach
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(a, y_train)

X_test_vectorized = vect.transform(X_test['Review_Text'])
b = X_test_vectorized.toarray()

predict1 = clf.predict(b)
cm2 = confusion_matrix(y_test, predict1)
#accuracy using naivebayes is 0.75


##############------------------------###########################
##############Predicting best and worst hotels and reviews#############

#probab contains the confidence of prediction
probab = model.predict_proba(vect.transform(X_test['Review_Text']))
X_test_pred = np.array(X_pre_test)



sorted_prob = probab[:,2].argsort()
best_reviews = X_test_pred[sorted_prob[-10:]]
worst_reviews = X_test_pred[sorted_prob[:10]]

best_hotels = X_test_pred[sorted_prob[-10:]][:,0]
worst_hotels = X_test_pred[sorted_prob[:10]][:,0]






