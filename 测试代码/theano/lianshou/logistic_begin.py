#! /usr/bin/env python
#-*- coding: utf-8 -*-import numpy
import numpy as np
import theano
import theano.tensor as T
import timeit
import gzip
import pickle
import os


class logisticRegression(object):
    def __init__(self,n_in,n_out):
        w_value = np.zeros((n_in,n_out))
        b_value = np.zeros(n_out)
        self.w = theano.shared(value=w_value,name='w',allow_downcast=True)
        self.b = theano.shared(value=b_value,name='b',allow_downcast=True)
        self.params = [self.w,self.b]

    def _calcule(self,x):
        # y_out = T.nnet.nnet.sigmoid(T.dot(x,self.w)+self.b)
        y_out = T.nnet.softmax(T.dot(x,self.w)+self.b)
        return y_out

def _negative_log_likelihood(y,y_out):
    cost_function = -T.mean(T.log(y_out)[T.arange(y.shape[0]), y])
    return cost_function
def _errors(y,y_out):
    y_predict = T.argmax(y_out)
    return T.mean(T.neq(y_predict, y))
def _train(n_in,n_out,lr,n_epochs,batch_size):
    index = T.lscalar()  # index to a [mini]batch
    f = gzip.open("mnist.pkl.gz",'rb')
    train_set, valid_set, test_set = pickle.load(f)
    train_set_x,train_set_y = train_set
    valid_set_x, valid_set_y = valid_set
    test_set_x, test_set_y = train_set
    n_train_batches = len(train_set_x) // batch_size
    n_valid_batches = len(valid_set_x) // batch_size
    n_test_batches = len(test_set_x) // batch_size

    floatX = 'float32'
    train_set_x = theano.shared(np.asarray(train_set_x,dtype=theano.config.floatX))
    train_set_y = theano.shared(np.asarray(train_set_y, dtype=theano.config.floatX))
    valid_set_x = theano.shared(np.asarray(valid_set_x, dtype=theano.config.floatX))
    valid_set_y = theano.shared(np.asarray(valid_set_y, dtype=theano.config.floatX))
    test_set_x = theano.shared(np.asarray(test_set_x, dtype=theano.config.floatX))
    test_set_y = theano.shared(np.asarray(test_set_y, dtype=theano.config.floatX))
    train_set_y = T.cast(train_set_y, 'int32')
    valid_set_y = T.cast(valid_set_y, 'int32')
    test_set_y = T.cast(test_set_y, 'int32')




    logistic = logisticRegression(784,10)
    x = T.matrix('x')
    y = T.ivector('y')
    y_out = logistic._calcule(x)
    cost = _negative_log_likelihood(y,y_out)
    errors = _errors(y,y_out)
    gparams = [T.grad(cost,param) for param in logistic.params]
    updates = [(param,param-lr*gparam) for (param,gparam) in zip(logistic.params,gparams)]
    train_model= theano.function(inputs=[index],outputs=errors,updates=updates,
                                 givens={
                                     x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                     y: train_set_y[index * batch_size: (index + 1) * batch_size]
                                 },
                                    allow_input_downcast=True)
    valid_model = theano.function(inputs=[index],outputs=errors,
                                  givens={
                                      x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                                      y: valid_set_y[index * batch_size: (index + 1) * batch_size]
                                  },
                                    allow_input_downcast=True)
    test_model = theano.function(inputs=[index], outputs=errors,
                                 givens={
                                     x: test_set_x[index * batch_size: (index + 1) * batch_size],
                                     y: test_set_y[index * batch_size: (index + 1) * batch_size]
                                 },
                                    allow_input_downcast=True)


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
    _train(784,10,0.01,10000,20)

