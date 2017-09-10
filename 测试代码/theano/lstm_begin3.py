#! /usr/bin/env python
#-*- coding: utf-8 -*-import numpy

import numpy as np
import pickle as pkl
import theano
import theano.tensor as T
import matplotlib as plt
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
from sklearn.model_selection import train_test_split
import timeit
import os
import sys

class LSTMlayer:
    def __init__(self,n_input,n_hidden,n_output):
        self.n_input = n_input
        self.n_hidden = n_hidden
        #input
        #srng = RandomStreams(seed=234)
        init_wi0 = np.array(np.random.uniform(low=-np.sqrt(1./n_input),high=(np.sqrt(1./n_input)) ,size=(n_input,n_hidden)),dtype=theano.config.floatX)
        init_wi1 = np.array(np.random.uniform(low=-np.sqrt(1. / n_input), high=(np.sqrt(1. / n_input)), size=(n_hidden,n_hidden)),dtype=theano.config.floatX)
        init_wi2 = np.array(np.random.uniform(low=-np.sqrt(1. / n_input), high=(np.sqrt(1. / n_input)), size=(n_hidden,n_hidden)),dtype=theano.config.floatX)
        init_bi = np.zeros(shape=(n_hidden,),dtype=theano.config.floatX)
        self.wi0 = theano.shared(value=init_wi0,name='wi0')
        self.wi1 = theano.shared(value=init_wi1, name='wi1')
        self.wi2 = theano.shared(value=init_wi2, name='wi2')
        self.bi = theano.shared(value=init_bi,name='bi')

        #forget
        init_wf0 = np.array(np.random.uniform(low=-np.sqrt(1./n_input),high=(np.sqrt(1./n_input)) ,size=(n_input,n_hidden)),dtype=theano.config.floatX) #dot遵循正常矩阵的乘法
        init_wf1 = np.array(np.random.uniform(low=-np.sqrt(1. / n_input), high=(np.sqrt(1. / n_input)), size=(n_hidden,n_hidden)),dtype=theano.config.floatX)
        init_wf2 = np.array(np.random.uniform(low=-np.sqrt(1. / n_input), high=(np.sqrt(1. / n_input)), size=(n_hidden,n_hidden)),dtype=theano.config.floatX)
        init_bf = np.zeros(shape=(n_hidden,),dtype=theano.config.floatX)
        self.wf0 = theano.shared(value=init_wf0,name='wf0')
        self.wf1 = theano.shared(value=init_wf1, name='wf1')
        self.wf2 = theano.shared(value=init_wf2, name='wf2')
        self.bf = theano.shared(value=init_bf,name='bf')
        #cell
        init_wc0 = np.array(np.random.uniform(low=-np.sqrt(1./n_input),high=(np.sqrt(1./n_input)) ,size=(n_input,n_hidden)),dtype=theano.config.floatX)
        init_wc1 = np.array(np.random.uniform(low=-np.sqrt(1. / n_input), high=(np.sqrt(1. / n_input)), size=(n_hidden,n_hidden)),dtype=theano.config.floatX)
        init_wc2 = np.array(np.random.uniform(low=-np.sqrt(1. / n_input), high=(np.sqrt(1. / n_input)), size=(n_hidden,n_hidden)),dtype=theano.config.floatX)
        init_bc = np.zeros(shape=(n_hidden,),dtype=theano.config.floatX)
        self.wc0 = theano.shared(value=init_wc0,name='wc0')
        self.wc1 = theano.shared(value=init_wc1, name='wc1')
        self.wc2 = theano.shared(value=init_wc2, name='wc2')
        self.bc = theano.shared(value=init_bc,name='bc')

        #output
        init_wo0 = np.array(np.random.uniform(low=-np.sqrt(1./n_input),high=(np.sqrt(1./n_input)) ,size=(n_input,n_hidden)),dtype=theano.config.floatX)
        init_wo1 = np.array(np.random.uniform(low=-np.sqrt(1. / n_input), high=(np.sqrt(1. / n_input)), size=(n_hidden,n_hidden)),dtype=theano.config.floatX)
        init_wo2 = np.array(np.random.uniform(low=-np.sqrt(1. / n_input), high=(np.sqrt(1. / n_input)), size=(n_hidden,n_hidden)),dtype=theano.config.floatX)
        init_bo = np.zeros(shape=(n_hidden,),dtype=theano.config.floatX)
        self.wo0 = theano.shared(value=init_wo0,name='wo0')
        self.wo1 = theano.shared(value=init_wo1, name='wo1')
        self.wo2 = theano.shared(value=init_wo2, name='wo2')
        self.bo = theano.shared(value=init_bo,name='bo')

        self.params1 = [self.wi0,self.wi1,self.wi2,self.bi,self.wf0,self.wf1,self.wf2,self.bf,self.wc0,self.wc1,self.wc2,self.bc]

        #classifier
        init_w = np.array(np.random.uniform(low=-np.sqrt(1./n_input),high=np.sqrt(1./n_input),size=(n_hidden,n_output)),dtype=theano.config.floatX)
        init_b = np.array(np.random.uniform(low=-np.sqrt(1./n_input),high=np.sqrt(1./n_input),size=(n_output)),dtype=theano.config.floatX)
        self.w = theano.shared(value=init_w,name='w')
        self.b = theano.shared(value=init_b,name='b')
        self.params2 = [self.w,self.b]

        self.params = self.params1 + self.params2

        def _cellcalculate(x,bh,sc):                                 #bh和sc需要自己定义初值
            ai = T.dot(self.wi0,x)+T.dot(self.wi1,bh)+T.dot(self.wi2,sc)
            _bi = T.nnet.hard_sigmoid(ai+self.bi)

            af = T.dot(self.wf0,x)+T.dot(self.wf1,bh)+T.dot(self.wf2,sc)
            _bf = T.nnet.hard_sigmoid(af+self.bf)

            ac = T.dot(self.wc0,x)+T.dot(self.wc1,bh)+T.dot(self.wc2,sc)
            _bc = T.tanh(ac+self.bc)
            sc = _bf*sc + _bi*_bc

            ao = T.dot(self.wo0,x)+T.dot(self.wo1,bh)+T.dot(self.wo2,sc)
            _bo = T.nnet.hard_sigmoid(ao+self.bo)

            bh = _bo*(T.tanh(sc))

            return [bh,sc]

        self.x = T.matrix('x')
        self.y = T.ivector('y')
        self.lr = T.scalar('lr')
        [result_b, result_s], updates = theano.scan(_cellcalculate,
                                                truncate_gradient=-1,
                                                sequences=self.x,
                                                outputs_info=[np.zeros(self.n_hidden),np.zeros(self.n_hidden)])  # 全都初始化为0,不知道对不对.

        classiffier = T.nnet.softmax(T.dot(result_b,self.w)+self.b)
        _predict = T.argmax(classiffier)
        self.predict = theano.function(
            inputs=[self.x],
            outputs=_predict,
            allow_input_downcast=True
        )

        self._negative_log_likelihood = -T.mean(T.log(classiffier)[T.arange(self.y.shape[0]),self.y])

        self.gparams =T.grad(self._negative_log_likelihood,self.params)
        updates = [(params,params-self.lr*gparam)for params,gparam in zip(self.params,self.gparams)]
        self.negative_log_likehood = theano.function(
            inputs = [self.x,self.y,self.lr],
            outputs = self._negative_log_likelihood,
            updates = updates,
            allow_input_downcast=True
        )
        _precision = 1. - T.mean(T.neq(_predict,self.y))
        self.precision = theano.function(
            inputs=[self.x,self.y],
            outputs=_precision,
            allow_input_downcast=True
        )


def sgd_optimization(lr=1,n_epochs=100,filemane='data.pkl',batch_size=1,n_input=2,n_hidden=2,n_output=2):

    train_x=np.array([[1,2],
                       [4,5]])
    train_y =np.array([0,0])
    print('...building the model')
    classifier = LSTMlayer(
        n_input=n_input,
        n_hidden=n_hidden,
        n_output=n_output
    )
    updates = [
        (params,params-lr*gparams)for params,gparams in zip(classifier.params,classifier.gparams)
    ]
    train_model = theano.function(
        inputs=[classifier.x,classifier.y],
        outputs=classifier._negative_log_likelihood,
        updates = updates,
        allow_input_downcast=True
    )

    epoche=0
    while epoche<n_epochs:
        epoche = epoche+1
        train_model(train_x,train_y)
        print('epoch %i, cost%f' ,epoche,train_model(train_x,train_y))

if __name__ == '__main__':
    sgd_optimization()











































