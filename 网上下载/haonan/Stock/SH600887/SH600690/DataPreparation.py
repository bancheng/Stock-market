# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 10:27:02 2017

@author: Administrator
"""

from featureConstruction import featureConstruction

from littleFun import autoNorm

#import theano.tensor as T
import numpy as np 



# 准备好已经经过KNN筛选的训练集，以及选取模型的训练集，测试集
# 中国股市对应标签为 开，高，收，低，量，额
#外国股市对应标签为量，开，收，高，低，额

def loadData():
    dataSetReverse = np.load('600690sh060104170131.npy')
    m,n = np.shape(dataSetReverse)
    dataSet = np.zeros((m,n))
    for i in range(m):
        dataSet[i]= dataSetReverse[m-i-1]
    return dataSet

def disLabelLL(dataSet):
    m,n =np.shape(dataSet)
    dataLabels=[]
    for i in range(0,m-1):
        if (dataSet[i,3]<=dataSet[i+1,3]):
            dataLabels.append(1)
        else: dataLabels.append(0)# the last day doesn't have a label
    return dataLabels
    
    
def disLabelHH(dataSet):
    m,n =np.shape(dataSet)
    dataLabels=[]
    for i in range(0,m-1):
        if (dataSet[i,1]<=dataSet[i+1,1]):
            dataLabels.append(1)
        else: dataLabels.append(0)# the last day doesn't have a label
    return dataLabels    
    
def disLabelCC(dataSet):
    m,n =np.shape(dataSet)
    dataLabels=[]
    for i in range(0,m-1):
        if (dataSet[i,2]<=dataSet[i+1,2]):
            dataLabels.append(1)
        else: dataLabels.append(0)# the last day doesn't have a label
    return dataLabels
    

    
def dataPreLHC():
    dataSetR = loadData()
#    print dataSetR.shape[1]
    dataLabelsCC = disLabelCC(dataSetR)
    dataLabelsLL = disLabelLL(dataSetR)
    dataLabelsHH = disLabelHH(dataSetR)
    dataSetF = featureConstruction(dataSetR)


    
    dataSets = np.asarray(dataSetF)
    dataSets, ranges, minVals = autoNorm(dataSets)
    # 去掉前20个计算产生的不准确数值以及数据集中最后一个没有标签的量
    dataSets = dataSets[20:(-1)]
    dataLabelsLL = dataLabelsLL[20:]
    dataLabelsHH = dataLabelsHH[20:]
    dataLabelsCC = dataLabelsCC[20:]
    
    if (len(dataLabelsLL)!=dataSets.shape[0])or(len(dataLabelsHH)!=dataSets.shape[0])or(len(dataLabelsCC)!=dataSets.shape[0]):
        print  'Error',len(dataLabelsLL),len(dataLabelsHH),len(dataLabelsCC),dataSets.shape[0]
    return dataSets,dataLabelsLL,dataLabelsHH,dataLabelsCC
    

# 将数据集分成90%，5%，5%三份    
def splitData(dataSet,dataLabels,trainRate=0.9,validRate=0.05,testRate=0.05):
    m,n = np.shape(dataSet)
#    dataSet = theano.shared(np.asarray(dataSet,dtype=theano.config.floatX),borrow=True)
#    dataLabels = theano.shared(np.asarray(dataLabels,dtype=theano.config.floatX),borrow=True)
    trainNum = int(m*trainRate)
    validNum = int(m*validRate)
#    testNum = m-trainNum-validNum
    trainSet = dataSet[:trainNum]
    validSet = dataSet[trainNum:(trainNum+validNum)]
    testSet = dataSet[(trainNum+validNum):]
    trainLabels = dataLabels[:trainNum]
    validLabels = dataLabels[trainNum:(trainNum+validNum)]
    testLabels = dataLabels[(trainNum+validNum):]
    return trainSet,validSet,testSet,trainLabels,validLabels,testLabels


    
    
def finalPre():
    dataSets,dataLabelsLL,dataLabelsHH,dataLabelsCC = dataPreLHC()
     #划分数据集--LL情况下    
    trainSetLL,validSetLL,testSetLL,trainLabelLL,validLabelLL,testLabelLL = splitData(dataSets,dataLabelsLL,trainRate=0.9,validRate=0.05,testRate=0.05)
    trainSetHH,validSetHH,testSetHH,trainLabelHH,validLabelHH,testLabelHH = splitData(dataSets,dataLabelsHH,trainRate=0.9,validRate=0.05,testRate=0.05)
    trainSetCC,validSetCC,testSetCC,trainLabelCC,validLabelCC,testLabelCC = splitData(dataSets,dataLabelsCC,trainRate=0.9,validRate=0.05,testRate=0.05)    
    
    return trainSetLL,validSetLL,testSetLL,trainLabelLL,validLabelLL,testLabelLL,trainSetHH,validSetHH,testSetHH,trainLabelHH,validLabelHH,testLabelHH,trainSetCC,validSetCC,testSetCC,trainLabelCC,validLabelCC,testLabelCC
    
