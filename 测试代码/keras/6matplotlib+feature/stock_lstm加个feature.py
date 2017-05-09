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
    print (X_train.shape)

    i=0
    j=0
    while i<len(y_train):
        if(y_train[i]==0):
            j=j+1
        i=i+1
    print (j)
    print (i)
    print (float(j)/i)

    # x = np.arange(len(y_test))
    # plt.plot(x,y_test)
    # plt.show()

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
    # def maximum(a):
    #     b=np.zeros(a.size[0])
    #     i=0
    #     j=0
    #     while i<a.size[0]:
    #         c=0
    #         while j<a.size[1]:
    #             c=c+a[i,j]
    #         b[i]=c
    #     return b

    model = Sequential()
    model.add(LSTM(32,input_shape=(timestep, 4),return_sequences=True))
    # model.add(LSTM(32,return_sequences=True))
    # model.add(LSTM(16,return_sequences=True))
    model.add(LSTM(8))
    # model.add(Dense(1, activation=maximum))
    # print('Build model...')
    # model = Sequential()
    # model.add(Embedding(max_features, 256))
    # model.add(LSTM(256, 128)) # try using a GRU instead, for fun
    model.add(Dense(1,activation='sigmoid'))
    print ("model")

    #
    # # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
    #
    print("Train...")
    model.fit(X_train, y_train, batch_size=8, nb_epoch=1, validation_split=0.2,show_accuracy=True)
    score = model.evaluate(X_test, y_test, batch_size=8, show_accuracy=True)
    print('Test score:', score)
    proba = model.predict_classes(X_test, batch_size=8)
    # print (proba)
    x = np.arange(len(proba))
    plt.plot(x,proba)
    plt.show()

if __name__ == '__main__':
    train(14)
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
