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
    print _values.shape
    i=1
    datax = _values[0:timestep,2:8]
    print datax.shape
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
    output = open('data.pkl','wb')
    pickle.dump(datax,output,True)
    pickle.dump(datay,output,True)
    output.close()
    return

if __name__ == '__main__':
    chanshengdata("SH600089xiugai.csv",50)
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


