#! /usr/bin/env python
#-*- coding: utf-8 -*-import numpy
#加载相应的包
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import pandas as pd

import datetime

start = datetime.datetime(2008, 1, 1)

end = datetime.datetime(2016, 10, 27)

df = web.DataReader("000001.sz", 'yahoo', start, end)
#print f
df.plot()
plt.show()


