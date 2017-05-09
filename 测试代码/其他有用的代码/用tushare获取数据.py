#! /usr/bin/env python
#-*- coding: utf-8 -*-import numpy
import tushare as ts
import matplotlib.pyplot as plt

df = ts.get_hist_data('000001', start='2011-04-01', end='2016-06-18')
print df
# 所有的结果汇图
df.plot()
# 只将stock最高值进行汇图
df.high.plot()
plt.show()
