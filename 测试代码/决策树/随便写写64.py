#! /usr/bin/env python
#-*- coding: utf-8 -*-import numpy

import theano
import theano.tensor as T
import numpy as np


def _cellcalculate( x,bh, sc):  # bh和sc需要自己定义初值
    ai = T.dot(1., x) + T.dot(1., bh) + T.dot(1., sc)
    _bi = T.nnet.hard_sigmoid(ai + 1)

    af = T.dot(1., x) + T.dot(1., bh) + T.dot(1., sc)
    _bf = T.nnet.hard_sigmoid(af + 1.)

    ac = T.dot(1., x) + T.dot(1., bh) + T.dot(1., sc)
    _bc = T.tanh(ac + 1.)
    sc = _bf * sc + _bi * _bc

    ao = T.dot(1., x) + T.dot(1., bh) + T.dot(1., sc)
    _bo = T.nnet.hard_sigmoid(ao + 1.)

    bh = _bo * (T.tanh(sc))

    return bh,sc

    # outputs_info = T.as_tensor_variable(np.asarray(0, 1.dtype))  (如果outputs_info报错，可以试试这个）


x = T.matrix('x')
k = T.iscalar('k')
#A = T.vector('A')
#B = T.vector('B')
#print (input)
[result_b,result_s], updates = theano.scan(fn=_cellcalculate,
                                            truncate_gradient=-1,
                                            sequences=x,
                                            outputs_info=(np.array([0.,0.,0.,0.]),np.array([0.,0.,0.,0.]))
                                           )  # 全都初始化为0,不知道对不对.

predict = theano.function(inputs=[x],outputs=result_b)  # 1需不需要用中括号？
#predict1 = predict[input]
a=np.array([[1,2,3,4],[1,2,3,4]])
print predict(a)
