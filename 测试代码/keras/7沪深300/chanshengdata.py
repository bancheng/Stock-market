#! /usr/bin/env python
#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import cPickle as pickle
import csv

def chanshengdata(filename,timestep):
    df = pd.read_csv(filename)
    #print type(df)
    _values = df.values
    #print type(_values)
    i = 2
    while i<8:
        _max = max(_values[:,i])
        _min = min(_values[:,i])
        j=0
        while j<len(_values):
            _values[j,i]=_values[j,i]/(_max - _min)
            j = j+1
        i = i+1





    print _values.shape
    i=1
    datax = _values[0:timestep,2:8]
    print datax.shape


    print ("处理1")
    while i<=(_values.shape[0]-(timestep+1)):
        datax = np.vstack((datax,_values[i:i+timestep,2:8]))
        i=i+1
    datax = datax.reshape(datax.shape[0]/timestep,timestep,6)
    print datax.shape
    _datay = _values[timestep:_values.shape[0],8]
    print _datay
    k = 0
    datay=[]
    while k< _datay.size:
        if _datay[k]<0:
            datay.append(0)
        else:
            datay.append(1)
        k=k+1
    datay = np.array(datay)
    print datay.shape


    print ("处理2")
    Testnumber = int(0.2*len(datay))
    _rand = np.random.randint(len(datay),size=len(datay))
    X_test = datax[_rand[0]]
    y_test = datay[_rand[0]]
    X_train = datax[_rand[Testnumber]]
    y_train = datay[_rand[Testnumber]]
    i=1
    while i<Testnumber:
        X_test = np.vstack((X_test,datax[_rand[i]]))
        y_test = np.vstack((y_test,datay[_rand[i]]))
        i=i+1
    X_test = X_test.reshape(X_test.shape[0]/timestep,timestep,6)
    i=Testnumber+1
    while (i>(Testnumber-1)) & (i<len(datay)):
        X_train = np.vstack((X_train, datax[_rand[i]]))
        y_train = np.vstack((y_train, datay[_rand[i]]))
        i=i+1
    X_train = X_train.reshape(X_train.shape[0]/timestep,timestep,6)




    output = open('data.pkl','wb')
    pickle.dump(X_test,output,True)
    pickle.dump(X_train,output,True)
    pickle.dump(y_test, output, True)
    pickle.dump(y_train, output, True)
    output.close()
    return

if __name__ == '__main__':
    chanshengdata("SH600288.csv",50)
#_datax = _values[0:_values.shape[0]-int(_values.shape[0]%50),2:8]
##print _datax.shape
#datax = _datax.reshape(int(_values.shape[0]/50),50,6)
#print datax.shape
#
#i=1
#_datay = np.zeros(_values.shape[0]/50)
#while 50*i<_values.shape[0]:
#    _datay[i-1] = _values[50*i][8]
#    i=i+1
#print _datay
#datay = []
#j = 0
#while j<len(_datay):
#    if _datay[j]<0:
#        datay.append(0)
#    else:
#        datay.append(1)
#    j=j+1
#datay = np.array(datay)
#print datay


