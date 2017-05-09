#! /usr/bin/env python
#-*- coding: utf-8 -*-import numpy

__author__ = 'Administrator'
import theano
import theano.tensor as T
import numpy as np
A = T.vector("A")
B = T.vector("B")
def step(a,b,c,d):
    return a + c, b + d
outputs_info = T.as_tensor_variable(np.asarray(0,A.dtype))
[result_mul,result_test],updates=theano.scan(step,sequences=[A,B],outputs_info = [outputs_info , outputs_info])
result_mul = theano.function([A,B],[result_mul,result_test])
m = result_mul([4,2,1,1],[1,2,3,4])
print m[1]