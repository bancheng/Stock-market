#! /usr/bin/env python
#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import cPickle as pickle
import csv

def calculate(filename,timestep):
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
    line15 = _values[:, 3] - _values[:, 0]
    __values = np.zeros((_values.shape[0], 1))
    _values = np.hstack((_values, __values))
    i = 0
    while i < _values.shape[0]:
        _values[i, (_values.shape[1] - 1)] = line15[i]
        i = i + 1
    print line15

    _values = np.hstack((_values, MA10))
    _values = np.hstack((_values, MTM))
    _values = np.hstack((_values, WMA10))
    _values = np.hstack((_values, CCI))

    i = 1
    datax = _values[0:timestep, 0:(_values.shape[1]) ]
    print datax.shape

    print ("处理7")
    while i <= (_values.shape[0] - (timestep + 1)):
        datax = np.vstack((datax, _values[i:i + timestep, 0:(_values.shape[1])]))
        i = i + 1
    datax = datax[:,7:11]
    datax = datax.reshape(datax.shape[0] / timestep, timestep, 4)
    print datax.shape

    _datay = line15[timestep:_values.shape[0]]
    print _datay
    k = 0
    datay = []
    while k < _datay.size:
        if _datay[k] < 0:
            datay.append(0)
        else:
            datay.append(1)
        k = k + 1
    datay = np.array(datay)
    print datay.shape

    print ("处理8")
    Testnumber = int(0.2 * len(datay))
    _rand = np.random.randint(len(datay), size=len(datay))
    X_test = datax[_rand[0]]
    y_test = datay[_rand[0]]
    X_train = datax[_rand[Testnumber]]
    y_train = datay[_rand[Testnumber]]
    i = 1
    while i < Testnumber:
        X_test = np.vstack((X_test, datax[_rand[i]]))
        y_test = np.vstack((y_test, datay[_rand[i]]))
        i = i + 1
    X_test = X_test.reshape(X_test.shape[0] / timestep, timestep, (4))
    i = Testnumber + 1
    while (i > (Testnumber - 1)) & (i < len(datay)):
        X_train = np.vstack((X_train, datax[_rand[i]]))
        y_train = np.vstack((y_train, datay[_rand[i]]))
        i = i + 1
    X_train = X_train.reshape(X_train.shape[0] / timestep, timestep, (4))

    output = open('data.pkl', 'wb')
    pickle.dump(X_test, output, True)
    pickle.dump(X_train, output, True)
    pickle.dump(y_test, output, True)
    pickle.dump(y_train, output, True)
    output.close()
    return


if __name__ == '__main__':
    calculate("taiping2008-2016.csv", 14)
















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


