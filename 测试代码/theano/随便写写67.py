#! /usr/bin/env python
#-*- coding: utf-8 -*-import numpy

import numpy as np
import pickle as pkl
import theano
import theano.tensor as T
import matplotlib as plt
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
from sklearn.cross_validation import train_test_split
import timeit
import os
import sys

class LSTMlayer:
    def __init__(self,input,n_input,n_hidden):
        self.input = input
        self.n_input = n_input
        self.n_hidden = n_hidden
        #input
        #srng = RandomStreams(seed=234)
        init_wi0 = np.array(np.random.uniform(low=np.sqrt(1./n_input),high=(np.sqrt(1./n_input)) ,size=(n_input,n_hidden)),dtype=theano.config.floatX)
        init_wi1 = np.array(np.random.uniform(low=np.sqrt(1. / n_input), high=(np.sqrt(1. / n_input)), size=(n_input,n_hidden)),dtype=theano.config.floatX)
        init_wi2 = np.array(np.random.uniform(low=np.sqrt(1. / n_input), high=(np.sqrt(1. / n_input)), size=(n_input,n_hidden)),dtype=theano.config.floatX)
        init_bi = np.zeros(shape=(n_hidden,),dtype=theano.config.floatX)
        self.wi0 = theano.shared(value=init_wi0,name='wi0')
        self.wi1 = theano.shared(value=init_wi1, name='wi1')
        self.wi2 = theano.shared(value=init_wi2, name='wi2')
        self.bi = theano.shared(value=init_bi,name='bi')

        #forget
        init_wf0 = np.array(np.random.uniform(low=np.sqrt(1./n_input),high=(np.sqrt(1./n_input)) ,size=(n_input,n_hidden)),dtype=theano.config.floatX) #dot遵循正常矩阵的乘法
        init_wf1 = np.array(np.random.uniform(low=np.sqrt(1. / n_input), high=(np.sqrt(1. / n_input)), size=(n_input,n_hidden)),dtype=theano.config.floatX)
        init_wf2 = np.array(np.random.uniform(low=np.sqrt(1. / n_input), high=(np.sqrt(1. / n_input)), size=(n_input,n_hidden)),dtype=theano.config.floatX)
        init_bf = np.zeros(shape=(n_hidden,),dtype=theano.config.floatX)
        self.wf0 = theano.shared(value=init_wf0,name='wf0')
        self.wf1 = theano.shared(value=init_wf1, name='wf1')
        self.wf2 = theano.shared(value=init_wf2, name='wf2')
        self.bf = theano.shared(value=init_bf,name='bf')
        #cell
        init_wc0 = np.array(np.random.uniform(low=np.sqrt(1./n_input),high=(np.sqrt(1./n_input)) ,size=(n_input,n_hidden)),dtype=theano.config.floatX)
        init_wc1 = np.array(np.random.uniform(low=np.sqrt(1. / n_input), high=(np.sqrt(1. / n_input)), size=(n_input,n_hidden)),dtype=theano.config.floatX)
        init_wc2 = np.array(np.random.uniform(low=np.sqrt(1. / n_input), high=(np.sqrt(1. / n_input)), size=(n_input,n_hidden)),dtype=theano.config.floatX)
        init_bc = np.zeros(shape=(n_hidden,),dtype=theano.config.floatX)
        self.wc0 = theano.shared(value=init_wc0,name='wc0')
        self.wc1 = theano.shared(value=init_wc1, name='wc1')
        self.wc2 = theano.shared(value=init_wc2, name='wc2')
        self.bc = theano.shared(value=init_bc,name='bc')

        #output
        init_wo0 = np.array(np.random.uniform(low=np.sqrt(1./n_input),high=(np.sqrt(1./n_input)) ,size=(n_input,n_hidden)),dtype=theano.config.floatX)
        init_wo1 = np.array(np.random.uniform(low=np.sqrt(1. / n_input), high=(np.sqrt(1. / n_input)), size=(n_input,n_hidden)),dtype=theano.config.floatX)
        init_wo2 = np.array(np.random.uniform(low=np.sqrt(1. / n_input), high=(np.sqrt(1. / n_input)), size=(n_input,n_hidden)),dtype=theano.config.floatX)
        init_bo = np.zeros(shape=(n_hidden,),dtype=theano.config.floatX)
        self.wo0 = theano.shared(value=init_wo0,name='wo0')
        self.wo1 = theano.shared(value=init_wo1, name='wo1')
        self.wo2 = theano.shared(value=init_wo2, name='wo2')
        self.bo = theano.shared(value=init_bo,name='bo')

        self.params1 = [self.wi0,self.wi1,self.wi2,self.bi,self.wf0,self.wf1,self.wf2,self.bf,self.wc0,self.wc1,self.wc2,self.bc]

        def _cellcalculate(x,bh,sc):                                 #bh和sc需要自己定义初值
            ai = T.dot(x,self.wi0)+T.dot(bh,self.wi1)+T.dot(sc,self.wi2)
            _bi = T.nnet.hard_sigmoid(ai+self.bi)

            af = T.dot(x,self.wf0)+T.dot(bh,self.wf1)+T.dot(sc,self.wf2)
            _bf = T.nnet.hard_sigmoid(af+self.bf)

            ac = T.dot(x,self.wc0)+T.dot(bh,self.wc1)+T.dot(sc,self.wc2)
            _bc = T.tanh(ac+self.bc)
            sc = _bf*sc + _bi*_bc

            ao = T.dot(x,self.wo0)+T.dot(bh,self.wo1)+T.dot(sc,self.wo2)
            _bo = T.nnet.hard_sigmoid(ao+self.bo)

            bh = _bo*(T.tanh(sc))

            return bh,sc

            # outputs_info = T.as_tensor_variable(np.asarray(0, x.dtype))  (如果outputs_info报错，可以试试这个）
        x = T.matrix('x')
        print (input)
        [result_b, result_s], updates = theano.scan(fn=_cellcalculate,
                                                truncate_gradient=-1,
                                                sequences=x,
                                                outputs_info=[np.zeros(n_hidden),np.zeros(n_hidden)])  # 全都初始化为0,不知道对不对.

        self.predict = theano.function(inputs=[x],outputs=result_b)  # x需不需要用中括号？
        self.predict1=self.predict(x)[-1]








x_train = ([[1,2,3],[2,3,4],[3,4,5]])

y_train = np.array([0, 1, 0])

test=LSTMlayer(x_train,3,3)
print (test.predict)