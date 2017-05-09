#! /usr/bin/env python
#-*- coding: utf-8 -*-import numpy

import numpy as np
import pickle as pkl
import theano
import theano.tensor as T
import matplotlib as plt
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
import timeit
import os
import sys



y = np.linspace(0, 1, 100)

x = np.append(np.sin(2 * np.pi * y), (-np.sin(2 * np.pi * y)))

z = np.column_stack((x, np.append(y, y))).astype(dtype=np.float32)

print(np.roll(z,1))
