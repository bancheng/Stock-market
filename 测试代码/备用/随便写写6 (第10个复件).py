#! /usr/bin/env python
#-*- coding: utf-8 -*-import numpy

import numpy
import pickle as pkl

from collections import OrderedDict

import glob
import os
import shutil
with open("/home/tangdongge/data/review_polarity 2.0/txt_sentoken/neg/cv672_27988.txt",'r') as ff:
    f=open("/home/tangdongge/data/强行复制/f2","a")
    f.write(ff.read().strip())
    ff.close()
    f.write('\n')