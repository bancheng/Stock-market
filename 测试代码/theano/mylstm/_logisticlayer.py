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
            cost = cost -T.log(p_y_given_x[0,y[i]])
        return cost/(2)
    def y_predict(self,x):
        _y_predict = []
        for i in np.arange(2):
            lastlayeroutput = self.lstmlayer.predict(x[i])
            p_y_given_x = T.nnet.softmax(T.dot(lastlayeroutput, self.w) + self.b)
            y_predict = T.argmax(p_y_given_x, axis=1)
            _y_predict.append(y_predict)
        return _y_predict
    def errors(self,x,y):
        y_predict = self.y_predict(x)
        return T.mean(T.neq(y_predict,y))


        # self.y = T.ivector('y')
        #
        # self.negative_log_likelihood = -T.mean(T.log(self.p_y_given_x)[T.arange(self.y.shape[0]),self.y])
        #
        # self.errors = T.mean(T.neq(self.y_predict,self.y))
