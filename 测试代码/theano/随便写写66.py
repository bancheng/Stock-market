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
        [result_b, result_s], updates = theano.scan(_cellcalculate,
                                                truncate_gradient=-1,
                                                sequences=x,
                                                outputs_info=[np.zeros(self.n_hidden),np.zeros(self.n_hidden)])  # 全都初始化为0,不知道对不对.

        predict1 = theano.function(inputs=[x],outputs=result_b)  # x需不需要用中括号？deprecated
        self.predict = predict1(self.input)[-1]

class logisticlayer:
    def __init__(self,input,n_input,n_hidden,n_output):

        self.input=input
        self.n_hidden=n_hidden
        self.n_output=n_output
        self.w=theano.shared(
            value=np.zeros(
                (n_hidden,n_output),
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


        self.lstmlayer = LSTMlayer(input=self.input,n_input=n_input,n_hidden=n_hidden)
        self.lastlayeroutput=self.lstmlayer.predict

        print (self.lastlayeroutput)

        self.params = self.lstmlayer.params1 + self.params2

        self.p_y_given_x=T.nnet.softmax(T.dot(self.lastlayeroutput,self.w)+self.b) #分成两类
        self.y_predict=T.argmax(self.p_y_given_x,axis=1)
    def negative_log_likelihood(self,y):
        return -T.mean(T.log(self.p_y_given_x)[T.arrange(y.shape[0]),y])

    def errors(self,y):
        if y.ndim !=self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y',y.type,'y_pred',self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred,y))
        else:
            raise NotImplementedError()


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
    train_y =[0,1,0]
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





    # n_train_batches=train_x.get_value(borrow=True).shape[0]//batch_size
    # n_valid_batches=valid_x.get_value(borrow=True).shape[0]//batch_size
    # n_test_batches=test_x.get_value(borrow=True).shape[0]//batch_size
    # n_train_batches=train_x.shape[0]//batch_size
    # n_valid_batches=valid_x.shape[0]//batch_size
    # n_test_batches=test_x.shape[0]//batch_size
    n_train_batches= 3
    n_valid_batches=3
    n_test_batches=3


    print('...building the model')

    index=T.scalar()
    x=T.matrix('x')
    y=T.matrix('y')

    rng=np.random.RandomState(1234)

    classifier = logisticlayer(
        input=x,
        n_input=n_input,
        n_hidden=n_hidden,
        n_output=n_output
    )

    cost = classifier.negative_log_likelihood(y)

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_x[index*batch_size:(index+1)*batch_size],
            y: test_y[index*batch_size:(index+1)*batch_size]
        }
    )

    valid_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_x[index*batch_size:(index+1)*batch_size],
            y: valid_y[index*batch_size:(index+1)*batch_size]
        }
    )

    gparams = [T.grad(cost,param)for param in classifier.params]
    updates = [
        (param,param - lr*gparams)
        for param,gparams in zip(classifier.params,gparams)
    ]

    train_model = theano.function(
        inputs = [index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_x[index * batch_size: (index + 1) * batch_size],
            y: train_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    print('... training')

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
    # found
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
    # go through this many
    # minibatche before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [valid_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if (
                                this_validation_loss < best_validation_loss *
                                improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)) )###file=sys.stderr



if __name__ == '__main__':
    sgd_optimization()











































