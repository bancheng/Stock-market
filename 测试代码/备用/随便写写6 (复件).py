#! /usr/bin/env python
#-*- coding: utf-8 -*-import numpy

import numpy
import theano
from theano import tensor

a = tensor.dscalar()
b = tensor.dscalar()
c= a+b
f = theano.function([a,b],c)
assert (4==f(2.5,2.5))