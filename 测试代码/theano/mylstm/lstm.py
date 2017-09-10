#! /usr/bin/env python
#-*- coding: utf-8 -*-import numpy

import numpy as np
import theano
import theano.tensor as T
import timeit
import os
import _logisticlayer
import init
def sgd_optimization(lr=0.01,n_epochs=100,filemane='data.pkl',batch_size=1,n_input=5,n_hidden=5,n_output=2):
    train_x=np.array([[[1,2,3,4,5],
             [1, 2, 3, 4, 5]],
                     [[1, 2, 3, 4, 5],
                     [1, 2, 3, 4, 5]]])
                     # [[1, 2, 3, 4, 5],
                     # [1, 2, 3, 4, 5]]

    train_y =np.array([0,1])
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

    x=T.tensor('x')
    y=T.ivector('y')

    rng=np.random.RandomState(1234)


    classifier = _logisticlayer.logisticlayer(
        n_input=n_input,
        n_hidden=n_hidden,
        n_output=n_output
    )
    cost = classifier.negative_log_likelihood(x,y)
    _grad = T.grad(cost,classifier.params)
    # updates = [(classifier.params,classifier.params - lr*_grad)]
    train_model = theano.function(
        inputs=[x,y],
        outputs=cost,
        # updates = updates,
        allow_input_downcast=True
    )
    print train_model(train_x,train_y)

if __name__ == '__main__':
    sgd_optimization()











































