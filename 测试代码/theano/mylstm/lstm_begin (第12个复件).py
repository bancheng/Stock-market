#! /usr/bin/env python
#-*- coding: utf-8 -*-import numpy

import numpy as np
import init
import theano
import theano.tensor as T


class logisticlayer:
    def __init__(self,n_input,n_hidden,n_output):
        self.n_hidden=n_hidden
        self.n_output=n_output
        self.w=theano.shared(
            value=np.zeros(
                (n_output,n_hidden),
                dtype=theano.config.floatX
            ),
            name='w',
            borrow=True
        )
        self.b=theano.shared(
            value=np.zeros(
                (n_output,),   #一直不明白n_output后面的逗号是什么意思
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        self.params2 = [self.w, self.b]
        self.lstmlayer = init.LSTMlayer(n_input=n_input,n_hidden=n_hidden)
        self.params = self.lstmlayer.params1 + self.params2
    def negative_log_likelihood(self,xx,y):
        cost = 0
        for i in np.arange(2):
            lastlayeroutput = self.lstmlayer.predict(xx[i])
            p_y_given_x = T.nnet.softmax(T.dot(self.w,lastlayeroutput)) ##+ self.b)
            cost = cost -T.log(p_y_given_x[y[i]])
        return p_y_given_x[y[0]]
train_x=np.array([[[1,2,3,4,5],
         [1, 2, 3, 4, 5]],
                 [[1, 2, 3, 4, 5],
                 [1, 2, 3, 4, 5]]])
train_y =np.array([0,1])
x = T.tensor3('x')
y = T.ivector('y')
xx = T.matrix("xx")
classifier = logisticlayer(
    n_input=5,
    n_hidden=5,
    n_output=2
)
# out = theano.function(
#     inputs=[x,y],
#     outputs=classifier.negative_log_likelihood(x,y),
#     allow_input_downcast=True
# )
# print out(train_x, train_y)
cost = 0
lastlayeroutput = classifier.lstmlayer.predict(xx)
inter = T.dot(classifier.w, lastlayeroutput)
p_y_given_x = T.nnet.softmax(T.dot(classifier.w, lastlayeroutput))  ##+ self.b)
cost = cost - T.log(p_y_given_x[y[0]])
out = theano.function(
    inputs=[xx],
    outputs=inter,
    allow_input_downcast=True
)
print out(train_x[0])