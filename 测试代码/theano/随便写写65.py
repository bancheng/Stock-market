#! /usr/bin/env python
#-*- coding: utf-8 -*-import numpy
import numpy as np
import theano
import theano.tensor as T
class LSTMlayer:
    def __init__(self,n_input,n_hidden):
        self.n_input = n_input
        self.n_hidden = n_hidden
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
        def _cellcalculate(x,bh,sc):
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

        self.x = T.matrix("x")
        [result_b, result_s], updates = theano.scan(_cellcalculate,
                                                truncate_gradient=-1,
                                                sequences=self.x,
                                                outputs_info=[np.zeros(self.n_hidden),np.zeros(self.n_hidden)])

        self._predict = result_b[-1]
lstm = LSTMlayer(5,5)
train_x=np.array([[1,2,3,4,5],
                       [1,2,3,4,5]])
# x = T.matrix("x")
out = theano.function(
        inputs=[lstm.x],
        outputs = lstm._predict
    )
print out(train_x)