from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import cPickle as pkl

from keras.preprocessing import sequence, text
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import theano

def train():
    X1 = np.ones(500*6).reshape(500,6)
    X2 = np.ones(1000*6).reshape(1000,6)
    X2 = 2*X2
    X_train = np.vstack((X1,X2))
    X_test = X_train
    X3 = np.zeros(1000)
    X4 = np.ones(500)
    y_train = np.hstack((X3,X4))
    y_test = y_train
    model = Sequential()
    model.add(Dense(6, input_shape=(6,),init='uniform',activation='sigmoid'))
    # print('Build model...')
    # model = Sequential()
    # model.add(Embedding(max_features, 256))
    # model.add(LSTM(256, 128)) # try using a GRU instead, for fun
    model.add(Dense(1,activation='sigmoid'))

    print ("model")

    #
    # # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=["accuracy"])
    #
    print("Train...")
    model.fit(X_train, y_train, batch_size=16, nb_epoch=1, validation_split=0.2,show_accuracy=True)
    score = model.evaluate(X_test, y_test, batch_size=16, show_accuracy=True)
    get_3rd_layer_output = theano.function([model.layers[0].input],
                                            model.layers[1].get_output(train=False))
    layer_output = get_3rd_layer_output(X_train)
    print('zhongjian',layer_output)
    print('Test score:', score)
    proba = model.predict_classes(X_train, batch_size=32)
    print (proba)
    x = np.arange(len(y_train))
    plt.plot(x,proba)
    plt.show()

if __name__ == '__main__':
    train()
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
