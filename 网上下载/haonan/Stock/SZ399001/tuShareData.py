# -*- coding: utf-8 -*-
#import tushare as ts
import numpy as np





'''
def loadData():
    df = ts.get_h_data('399001',index=True,start='1999-07-26',end = '2001-12-23')
    dataSet = df.get_values()
    # dataSet 中6列分别是开盘，最高，最低和收盘
    return dataSet
   
def loadDataWithData():
    df = ts.get_h_data('000001',index=True,start='1999-07-26',end = '2016-12-23')
    dataSetWithDate = df.get_values
    return dataSetWithDate
'''
def loadData():
    dataSetReverse = np.load('399001sz990726161223.npy')
    m,n = np.shape(dataSetReverse)
    dataSet = np.zeros((m,n))
    for i in range(m):
        dataSet[i]= dataSetReverse[m-i-1]
    return dataSet
