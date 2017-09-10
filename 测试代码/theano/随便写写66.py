#! /usr/bin/env python
#-*- coding: utf-8 -*-import numpy

import numpy as np
import pickle as pkl
import theano
import theano.tensor as T
import matplotlib as plt
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
from sklearn.model_selection import train_test_split
import timeit
import os
import sys