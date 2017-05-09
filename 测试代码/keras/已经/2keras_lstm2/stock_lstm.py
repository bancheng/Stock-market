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

def train(timestep):
    f=open("data.pkl",'rb')
    X_test = pkl.load(f)
    X_train = pkl.load(f)
    y_test = pkl.load(f)
    y_train = pkl.load(f)
    f.close()
    x = np.arange(len(y_test))


    # Testnumber = int(0.2*len(datay))
    # _rand = np.random.randint(len(datay),size=len(datay))
    # X_test = datax[_rand[0]]
    # y_test = datay[_rand[0]]
    # X_train = datax[_rand[Testnumber]]
    # y_train = datay[_rand[Testnumber]]
    # i=1
    # while i<Testnumber:
    #     X_test = np.vstack((X_test,datax[_rand[i]]))
    #     y_test = np.vstack((y_test,datay[_rand[i]]))
    #     i=i+1
    # X_test = X_test.reshape(X_test.shape[0]/timestep,timestep,6)
    # i=Testnumber+1
    # while (i>(Testnumber-1)) & (i<len(datay)):
    #     X_train = np.vstack((X_train, datax[_rand[i]]))
    #     y_train = np.vstack((y_train, datay[_rand[i]]))
    #     i=i+1
    # X_train = X_train.reshape(X_train.shape[0]/timestep,timestep,6)

    # print('X_train shape:', X_train.shape)
    # print('X_test shape:', X_test.shape)
    model = Sequential()
    model.add(LSTM(16, return_sequences=True,input_shape=(timestep, 6)))
    # print('Build model...')
    # model = Sequential()
    # model.add(Embedding(max_features, 256))
    # model.add(LSTM(256, 128)) # try using a GRU instead, for fun
    print ("model")
    model.add(Dropout(0.5))
    model.add(LSTM(16,return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(16))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    #
    # # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
    #
    print("Train...")
    model.fit(X_train, y_train, batch_size=16, nb_epoch=1, validation_split=0.2)
    score = model.evaluate(X_test, y_test, batch_size=16, show_accuracy=True)
    print('Test score:', score)
    proba = model.predict_classes(X_test, batch_size=32)
    print (proba)
    plt.plot(x,proba)
    plt.show()


if __name__ == '__main__':
    train(50)
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
