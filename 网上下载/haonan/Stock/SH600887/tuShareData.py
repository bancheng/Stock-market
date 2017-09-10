# -*- coding: utf-8 -*-
#import tushare as ts
import numpy as np

import tushare as ts



'''
def loadData():
    df = ts.get_h_data('000001',index=True,start='1999-07-26',end = '2001-12-23')
    dataSet = df.get_values()
    # dataSet 中6列分别是开盘，最高，最低和收盘
    return dataSet
   
def loadDataWithData():
    df = ts.get_h_data('000001',index=True,start='1999-07-26',end = '2016-12-23')
    dataSetWithDate = df.get_values
    return dataSetWithDate
'''
if __name__ == '__main__':
    df = ts.get_h_data('600887',index=False,start='2006-01-04',end = '2017-01-31')
    dataSet = df.get_values()
    np.save('600887sh060104170131.npy',dataSet)
    