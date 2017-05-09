import numpy as np
import cPickle as pkl
import pandas as pd

from keras.preprocessing import sequence, text
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.cross_validation import train_test_split

def train(filename):
    df = pd.read_csv(filename)
    _values = df.values
    _values = _values[:,1:_values.shape[1]]
    print _values.shape
    _xtrain = _values[0:(_values.shape[0]-1)]
    print _xtrain.shape
    line15 = _values[1:, 3] - _values[1:, 0]
    _ytrain = np.zeros(len(line15))
    i=0
    while(i<len(line15)):
        if(line15[i]<0):
            _ytrain[i]=0
        else:
            _ytrain[i] = 1
        i=i+1
    print _ytrain.shape

    x_train,x_test,y_train,y_test = train_test_split(_xtrain,_ytrain,test_size=0.2)
    print ('chuli1')
    clf = svm.SVC(kernel='linear').fit(_xtrain, _ytrain)
    print('chli2')
    print(clf)
    clf_linear = svm.SVC(kernel='linear').fit(x_train,y_train)
    print('chuli1')
    print(clf_linear.predict(x_train))
    ##clf_poly = svm.SVC(kenel = 'poly',degree=3).fit(x_train,y_train)
    return
if __name__ == '__main__':
    train('taiping2008-2016.csv')
#
# classes = model.predict_classes(X_test, batch_size=batch_size)
# acc = np_utils.accuracy(classes, y_test)
#
# print('Test accuracy:', acc)
#
# store_weights = {}
# for layer in model.layers :
#     store_weights[layer] = layer.get_weights()
#
# # create a new model of same structure minus last layers, to explore intermediate outputs
# print('Build truncated model')
# chopped_model = Sequential()
# chopped_model.add(Embedding(max_features, 256, weights=model.layers[0].get_weights()))
# chopped_model.add(LSTM(256, 128, weights=model.layers[1].get_weights()))
# chopped_model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")
#
# # pickle intermediate outputs, model weights
# train_activations = chopped_model.predict(X_train, batch_size=batch_size)
# test_activations = chopped_model.predict(X_test, batch_size=batch_size)
# outputs = dict(final=classes, acc=acc, weights=store_weights, y_train=y_train, y_test=y_test,
#     train_activations=train_activations, test_activations=test_activations)
#
# pkl.dump(outputs, open('results/predicted_activations.pkl', 'wb'),
#     protocol=pkl.HIGHEST_PROTOCOL)
