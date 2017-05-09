#! /usr/bin/env python
#-*- coding: utf-8 -*-import numpy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data
y = iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

estimator = DecisionTreeClassifier(max_leaf_nodes=3,random_state=0)
estimator.fit(X_train,y_train)

n_nodes = estimator.tree_.node_count
children_left = estimator.tree_.children_left
children_right = estimator.tree_.children_right
feature = estimator.tree_.feature
threshold = estimator.tree_.threshold

node_depth = np.zeros(shape=n_nodes)
is_leaves = np.zeros(shape=n_nodes,dtype=bool)
stack = [(0,-1)]

while len(stack)>0:
    node_id,parent_depth = stack.pop()
    node_depth[node_id] = parent_depth +1