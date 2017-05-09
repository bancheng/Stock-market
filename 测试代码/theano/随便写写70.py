#! /usr/bin/env python
#-*- coding: utf-8 -*-import numpy
import theano
import theano.tensor as T
import numpy as np

def fn(x,y):
    return x+y

x=T.matrix('x')
y=T.tensor3('y')
result,updates = theano.scan(
    fn=fn,
    sequences=x,
    outputs_info=np.zeros(5)
)

f=theano.function(
    inputs=[x],
    outputs=result
)

inter,updates = theano.scan(
    fn=f,
    sequences=y
)


test = theano.function(
    inputs=[y],
    outputs=inter
)





train_x = np.array([[[1,2,3,4,5],
                   [1,2,3,4,5]]])
print test(train_x)
