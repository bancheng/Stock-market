# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T



# 归一化函数
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals,(m,1))
    normDataSet = normDataSet/np.tile(ranges,(m,1))
    return normDataSet, ranges, minVals
    
def makeShared(a):
    shared_a = theano.shared(np.asarray(a,dtype=theano.config.floatX),borrow=True)
    return shared_a
    
    
def makeAllShared(trainSet,validSet,testSet,trainLabel,validLabel,testLabel):
    shared_train_set = makeShared(trainSet)
    shared_valid_set = makeShared(validSet)
    shared_test_set = makeShared(testSet)
    shared_trainLabels = makeShared(trainLabel)
    shared_validLabels = makeShared(validLabel)
    shared_testLabels = makeShared(testLabel)
    return shared_train_set,shared_valid_set,shared_test_set,T.cast(shared_trainLabels,'int32'),T.cast(shared_validLabels,'int32'),T.cast(shared_testLabels,'int32')    
    
def makeResultMat(PreLL,PreHH,CC):
    ResultMat = np.zeros((len(CC),3))
    for i in range(len(CC)):
        ResultMat[i][0] = PreLL[i]
        ResultMat[i][1] = PreHH[i]
        ResultMat[i][2] = CC[i]
    return ResultMat
    
def calNum(ResultMat):
    num = np.zeros((8,1))
    for i in range(ResultMat.shape[0]):
        if(ResultMat[i][0]==1)and(ResultMat[i][1]==1)and(ResultMat[i][2]==1):num[0]+=1
        elif((ResultMat[i][0]==1)and(ResultMat[i][1]==1)and(ResultMat[i][2]==0)):num[1]+=1
        elif((ResultMat[i][0]==1)and(ResultMat[i][1]==0)and(ResultMat[i][2]==1)):num[2]+=1
        elif((ResultMat[i][0]==1)and(ResultMat[i][1]==0)and(ResultMat[i][2]==0)):num[3]+=1
        elif((ResultMat[i][0]==0)and(ResultMat[i][1]==1)and(ResultMat[i][2]==1)):num[4]+=1
        elif((ResultMat[i][0]==0)and(ResultMat[i][1]==1)and(ResultMat[i][2]==0)):num[5]+=1
        elif((ResultMat[i][0]==0)and(ResultMat[i][1]==0)and(ResultMat[i][2]==1)):num[6]+=1
        elif((ResultMat[i][0]==0)and(ResultMat[i][1]==0)and(ResultMat[i][2]==0)):num[7]+=1
    numList=[]
    for i in [0,2,4,6]:
        numList.append([num[i][0],num[i+1][0]])  
    return numList
    
def calRight(validList,testList):
    if(validList[0][0]>=validList[0][1]):
        index11 = 0
    else:index11 = 1
    if(validList[1][0]>=validList[1][1]):
        index10 = 0
    else:index10 = 1
    if(validList[2][0]>=validList[2][1]):
        index01 = 0
    else:index01 = 1
    if(validList[3][0]>=validList[3][1]):
        index00 = 0
    else:index00 = 1
    return float(testList[0][index11]+testList[1][index10]+testList[2][index01]+testList[3][index00])





        
    
    
    
    
    