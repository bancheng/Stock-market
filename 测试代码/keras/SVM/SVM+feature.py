#! /usr/bin/env python
#-*- coding: utf-8 -*-
import numpy as np
import cPickle as pkl
import pandas as pd
from sklearn import svm
from sklearn.cross_validation import train_test_split
def train(filename):
    df = pd.read_csv(filename)
    #print type(df)
    _values = df.values
    #print type(_values)

    _values = _values[:,1:]

    ####
    print("处理")
    _values1 = np.zeros(_values.shape)
    j = 0
    i=0
    while(j<_values.shape[0]-1):
        if ((_values[j,4]==0)&(_values[j+1,4]==0)):
            i=i-1
        else:
            _values1[i] = _values[j]
        i=i+1
        j = j+1

    _values = _values1[0:i-1]

    ####
    print('step1')

    ###MA10
    def MA(n):
        MA10 = np.zeros((_values.shape[0], 1))
        i = n - 1
        j = 0
        while i < _values.shape[0]:
            while j < n:
                MA10[i] = MA10[i] + _values[i - n + 1 + j, 3]  ##__values[j,3]是闭盘价
                j = j + 1
            i = i + 1
        return MA10

    MA10 = MA(10)

    print('step2')
    ###WMA10
    WMA10 = np.zeros((_values.shape[0], 1))
    i = 9
    j = 0
    while i < _values.shape[0]:
        while j < 10:
            WMA10[i] = WMA10[i] + (j + 1) * _values[i - 9 + j, 3]  ##__values[j,3]是闭盘价
            j = j + 1
        WMA10[i] = (WMA10[i]) / 55
        i = i + 1

    print('step3')
    ###MTM
    MTM = np.zeros((_values.shape[0], 1))
    i = 9
    while i < _values.shape[0]:
        MTM[i] = _values[i, 3] - _values[i - 9, 3]
        i = i + 1
    print('step4')

    ###CCI
    def TP(t):
        TP = (_values[t, 1] + _values[t, 2] + _values[t, 3]) / 3
        return TP

    def TP14(t):
        TP14 = 0
        i = 0
        while i < 14:
            TP14 = TP14 + TP(t - i)
            i = i + 1
        TP14 = TP14 / 14
        return TP14

    def Deviation14(t):
        Deviation14 = 0
        i = 0
        while i < 14:
            Deviation14 = Deviation14 + (TP(t - i) - TP14(t))
            i = i + 1
        Deviation14 = Deviation14 / 14
        return Deviation14

    CCI = np.zeros((_values.shape[0], 1))
    i = 13
    while i < _values.shape[0]:
        CCI[i] = (TP(i) - TP14(i)) / (0.015 * Deviation14(i) + 0.00001)
        i = i + 1

    ###
    print('step6')
    ### chanshengdata
    line15 = _values[1:, 3] - _values[1:, 0]
    i = 0
    _ytrain = np.zeros(len(line15)-14)
    i=0
    while(i<(len(line15)-14)):
        if(line15[i]<0):
            _ytrain[i]=0
        else:
            _ytrain[i] = 1
        i=i+1
    print _ytrain.shape

    _values = np.hstack((_values, MA10))
    _values = np.hstack((_values, MTM))
    _values = np.hstack((_values, WMA10))
    _values = np.hstack((_values, CCI))

    _xtrain = _values[14:(_values.shape[0] - 1)]



    x_train,x_test,y_train,y_test = train_test_split(_xtrain[:,6:10],_ytrain,test_size=0.2)
    print x_train
    print x_train.shape,y_train.shape

    clf_linear = svm.SVC(kernel='rbf',C=0.1).fit(x_train,y_train)
    print('chuli1')
    print(clf_linear.predict(x_train))
    print(np.mean(clf_linear.predict(x_train) == y_train))
    print(np.mean(clf_linear.predict(x_test) == y_test))
    return

if __name__ == '__main__':
    train('taiping2008-2016.csv')











if __name__ == '__main__':
    train("taiping2008-2016.csv")
















    # _datax = _values[0:_values.shape[0]-int(_values.shape[0]%50),2:8]
    ##print _datax.shape
    # datax = _datax.reshape(int(_values.shape[0]/50),50,6)
    # print datax.shape
    #
    # i=1
    # _datay = np.zeros(_values.shape[0]/50)
    # while 50*i<_values.shape[0]:
    #    _datay[i-1] = _values[50*i][8]
    #    i=i+1
    # print _datay
    # datay = []
    # j = 0
    # while j<len(_datay):
    #    if _datay[j]<0:
    #        datay.append(0)
    #    else:
    #        datay.append(1)
    #    j=j+1
    # datay = np.array(datay)
    # print datay


