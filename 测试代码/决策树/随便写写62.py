#! /usr/bin/env python
#-*- coding: utf-8 -*-import numpy

import numpy as np
import pickle as pkl
import theano
import theano.tensor as T
import matplotlib as plt
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
import numpy
import pickle as pkl



a=np.array([[1,2,3],
            [4,5,6]])
b=np.array([[1,2,3],
            [4,5,6]]
            )
print(T.dot(a,b).eval())