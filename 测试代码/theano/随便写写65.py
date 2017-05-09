##! /usr/bin/env python
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

class LSTMlayer(object):
    def __init__(self,input,n_input,n_hidden,n_output):
        self.input = input
        self.n_input = n_input
        self.n_hidden = n_hidden
        #input
        #srng = RandomStreams(seed=234)
        init_wi0 = np.array(np.random.uniform(low=np.sqrt(1./n_input),high=(np.sqrt(1./n_input)) ,size=(n_input,n_hidden)),dtype=theano.config.floatX)
        init_wi1 = np.array(np.random.uniform(low=np.sqrt(1. / n_input), high=(np.sqrt(1. / n_input)), size=(n_hidden,n_hidden)),dtype=theano.config.floatX)
        init_wi2 = np.array(np.random.uniform(low=np.sqrt(1. / n_input), high=(np.sqrt(1. / n_input)), size=(n_hidden,n_hidden)),dtype=theano.config.floatX)
        init_bi = np.zeros(shape=(n_hidden,),dtype=theano.config.floatX)
        self.wi0 = theano.shared(value=init_wi0,name='wi0')
        self.wi1 = theano.shared(value=init_wi1, name='wi1')
        self.wi2 = theano.shared(value=init_wi2, name='wi2')
        self.bi = theano.shared(value=init_bi,name='bi')

        #forget
        init_wf0 = np.array(np.random.uniform(low=np.sqrt(1./n_input),high=(np.sqrt(1./n_input)) ,size=(n_input,n_hidden)),dtype=theano.config.floatX) #dot遵循正常矩阵的乘法
        init_wf1 = np.array(np.random.uniform(low=np.sqrt(1. / n_input), high=(np.sqrt(1. / n_input)), size=(n_hidden,n_hidden)),dtype=theano.config.floatX)
        init_wf2 = np.array(np.random.uniform(low=np.sqrt(1. / n_input), high=(np.sqrt(1. / n_input)), size=(n_hidden,n_hidden)),dtype=theano.config.floatX)
        init_bf = np.zeros(shape=(n_hidden,),dtype=theano.config.floatX)
        self.wf0 = theano.shared(value=init_wf0,name='wf0')
        self.wf1 = theano.shared(value=init_wf1, name='wf1')
        self.wf2 = theano.shared(value=init_wf2, name='wf2')
        self.bf = theano.shared(value=init_bf,name='bf')
        #cell
        init_wc0 = np.array(np.random.uniform(low=np.sqrt(1./n_input),high=(np.sqrt(1./n_input)) ,size=(n_input,n_hidden)),dtype=theano.config.floatX)
        init_wc1 = np.array(np.random.uniform(low=np.sqrt(1. / n_input), high=(np.sqrt(1. / n_input)), size=(n_hidden,n_hidden)),dtype=theano.config.floatX)
        init_wc2 = np.array(np.random.uniform(low=np.sqrt(1. / n_input), high=(np.sqrt(1. / n_input)), size=(n_hidden,n_hidden)),dtype=theano.config.floatX)
        init_bc = np.zeros(shape=(n_hidden,),dtype=theano.config.floatX)
        self.wc0 = theano.shared(value=init_wc0,name='wc0')
        self.wc1 = theano.shared(value=init_wc1, name='wc1')
        self.wc2 = theano.shared(value=init_wc2, name='wc2')
        self.bc = theano.shared(value=init_bc,name='bc')

        #output
        init_wo0 = np.array(np.random.uniform(low=np.sqrt(1./n_input),high=(np.sqrt(1./n_input)) ,size=(n_input,n_hidden)),dtype=theano.config.floatX)
        init_wo1 = np.array(np.random.uniform(low=np.sqrt(1. / n_input), high=(np.sqrt(1. / n_input)), size=(n_hidden,n_hidden)),dtype=theano.config.floatX)
        init_wo2 = np.array(np.random.uniform(low=np.sqrt(1. / n_input), high=(np.sqrt(1. / n_input)), size=(n_hidden,n_hidden)),dtype=theano.config.floatX)
        init_bo = np.zeros(shape=(n_hidden,),dtype=theano.config.floatX)
        self.wo0 = theano.shared(value=init_wo0,name='wo0')
        self.wo1 = theano.shared(value=init_wo1, name='wo1')
        self.wo2 = theano.shared(value=init_wo2, name='wo2')
        self.bo = theano.shared(value=init_bo,name='bo')

        self.params1 = [self.wi0,self.wi1,self.wi2,self.bi,self.wf0,self.wf1,self.wf2,self.bf,self.wc0,self.wc1,self.wc2,self.bc]

        self.w = theano.shared(
            value=np.zeros(
                (n_hidden, n_output),
                dtype=theano.config.floatX
            ),
            name='w',
            borrow=True
        )
        self.b = theano.shared(
            value=np.zeros(
                (n_output,),  # 一直不明白n_output后面的逗号是什么意思
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        self.params2 = [self.w, self.b]

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

            # outputs_info = T.as_tensor_variable(np.asarray(0, x.dtype))  (如果outputs_info报错，可以试试这个）

        x = T.matrix('x')
        #print (self.input)
        [result_b, result_s], updates = theano.scan(_cellcalculate,
                                                truncate_gradient=-1,
                                                sequences=x,
                                                outputs_info=[np.zeros(self.n_hidden),np.zeros(self.n_hidden)])  # 全都初始化为0,不知道对不对.

        self.predict1 = result_b[-1] # x需不需要用中括号





        self.params = self.params1 + self.params2

        self.p_y_given_x = T.nnet.softmax(T.dot(self.predict1, self.w) + self.b)  # 分成两类

        self.y_predict = T.argmax(self.p_y_given_x, axis=1)

    def negative_log_likelihood(self,y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
    def errors(self,y):
        return T.mean(T.neq(self.y_predict, y))



def sgd_optimization(lr=0.01,n_epochs=100,filemane='data.pkl',batch_size=1,n_input=5,n_hidden=10,n_output=2):
    # datasets=load_data(filemane)
    # train_x,train_y=datasets[0]
    # valid_x,valid_y=datasets[1]
    # test_x,test_y=datasets[2]
    #

    train_x=([[1,2,3,4,5],
                       [1, 2, 3, 4, 5]]
                     #  [[1, 2, 3, 4, 5],
                     #  [1, 2, 3, 4, 5]],
                     # [[1, 2, 3, 4, 5],
                     # [1, 2, 3, 4, 5]]
                      )
    train_y =([0,1,0])
    valid_x=([[1,2,3,4,5],
                       [1, 2, 3, 4, 5]]
                      # [[1, 2, 3, 4, 5],
                     #  [1, 2, 3, 4, 5]],
                     # [[1, 2, 3, 4, 5],
                     # [1, 2, 3, 4, 5]]
                      )
    valid_y =[0,1,0]
    test_x=([[1,2,3,4,5],
                       [1, 2, 3, 4, 5]]
                     #  [[1, 2, 3, 4, 5],
                     #  [1, 2, 3, 4, 5]],
                     # [[1, 2, 3, 4, 5],
                     # [1, 2, 3, 4, 5]]
                      )
    test_y =[0,1,0]




    print('...building the model')

    index=T.lscalar()
    x=T.matrix('x')
    y=T.ivector('y')

    rng=np.random.RandomState(1234)

    classifier = LSTMlayer(
        input=x,
        n_input=n_input,
        n_hidden=n_hidden,
        n_output=n_output
    )

    cost = classifier.negative_log_likelihood(y)

    gparams = [T.grad(cost,param)for param in classifier.params]
    updates = [
        (param,param - lr*gparams)
        for param,gparams in zip(classifier.params,gparams)
    ]

    train_model = theano.function(
        inputs = [train_x,train_y],
        outputs=cost,
        updates=updates
    #     givens={
    #         x: train_x[index * batch_size: (index + 1) * batch_size],
    #         y: train_y[index * batch_size: (index + 1) * batch_size]
    #     }
     )
    print('... training')

if __name__ == '__main__':
    sgd_optimization()












































