# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:28:47 2017

@author: Administrator
"""

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as RF


def SVM(C=1.0,kernel = 'rbf',random_state=1234):
    return svm.SVC(C=C,kernel=kernel)
    
def Random_forests(n_estimators,min_samples_split,random_state):
    return RF(n_estimators=n_estimators,min_samples_split = min_samples_split,random_state=random_state)