# -*- coding: utf-8 -*-
"""
Created on Sat May 06 19:50:26 2017

@author: Administrator
"""

from featureConstruction import featureConstruction

from littleFun import calNum,calRight,makeResultMat

#import theano.tensor as T
import numpy as np 
import operator
from SDA4 import test_SdA,predict

from DataPreparation import finalPre,splitData,loadData
from littleFun import autoNorm



def disLabelCC(dataSet):
    m,n =np.shape(dataSet)
    dataLabels=[]
    for i in range(0,m-1):
        if (dataSet[i,2]<=dataSet[i+1,2]):
            dataLabels.append(1)
        else: dataLabels.append(0)# the last day doesn't have a label
    return dataLabels
    
def dataPreCC():
    dataSetR = loadData()
#    print dataSetR.shape[1]
    dataLabelsCC = disLabelCC(dataSetR)

    dataSetF = featureConstruction(dataSetR)


    
    dataSets = np.asarray(dataSetF)
    dataSets, ranges, minVals = autoNorm(dataSets)
    # 去掉前20个计算产生的不准确数值以及数据集中最后一个没有标签的量
    dataSets = dataSets[20:(-1)]

    dataLabelsCC = dataLabelsCC[20:]

    if (len(dataLabelsCC)!=dataSets.shape[0]):
        print  len(dataLabelsCC),dataSets.shape[0]
    return dataSets,dataLabelsCC
    
def classify(inX, dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]



def seperateData(dataSets,dataLabels,k):
    m,n =np.shape(dataSets)
    rightData = []
    wrongData= []
    rightDataLabel = []
    wrongDataLabel = []
    for i in range(m):
        inx = dataSets[i,:]
        inxLabel = dataLabels[i]
        dataSet = np.concatenate((dataSets[:i],dataSets[(i+1):]))
        dataLabel = np.concatenate((dataLabels[:i],dataLabels[(i+1):]))
        inxLabelPre = classify(inx,dataSet,dataLabel,k)
        if(inxLabel==inxLabelPre):
            rightData.append(inx)
            rightDataLabel.append(inxLabel)
        else: 
            wrongData.append(inx)
            wrongDataLabel.append(inxLabel)
    rightData = np.asarray(rightData)
    rightDataLabel = np.asarray(rightDataLabel)
    wrongData = np.asarray(wrongData)
    wrongDataLabel = np.asarray(wrongDataLabel)
    return rightData,rightDataLabel,wrongData,wrongDataLabel



if __name__ == '__main__':
    print "对数据进行KNN处理后以LL,HH决定CC得到的实验结果:"
    trainSetLL,validSetLL,testSetLL,trainLabelLL,validLabelLL,testLabelLL,trainSetHH,validSetHH,testSetHH,trainLabelHH,validLabelHH,testLabelHH,trainSetCC,validSetCC,testSetCC,trainLabelCC,validLabelCC,testLabelCC = finalPre()

   
    bestRightRate = 0.0
    Results = []
    for k1 in  [3,5,7,9]:
        rightDataLL,rightDataLabelLL,wrongDataLL,wrongDataLabelLL =  seperateData(trainSetLL,trainLabelLL,k1)
#        for k2 in [3,5,7,9]:
#            rightData1LL,rightDataLabel1LL,wrongData1LL,wrongDataLabel1LL =  seperateData(wrongDataLL,wrongDataLabelLL,k2)
        newSetLL = rightDataLL
        newLabelLL = rightDataLabelLL
        for k3 in  [3,5,7,9]:
            rightDataHH,rightDataLabelHH,wrongDataHH,wrongDataLabelHH =  seperateData(trainSetHH,trainLabelHH,k3)
                
#                for k4 in  [3,5,7,9]:
#                    rightData1HH,rightDataLabel1HH,wrongData1HH,wrongDataLabel1HH =  seperateData(wrongDataHH,wrongDataLabelHH,k4)
                    
                        
                    

                        
            newSetHH = rightDataHH
            newLabelHH = rightDataLabelHH
            for n_hiddenL1 in [8,7,6]:
                for n_hiddenL2 in [5,4,3]:
                    test_SdA(newSetLL,validSetLL,testSetLL,newLabelLL,validLabelLL,testLabelLL,
                                                     finetune_lr=0.1, pretraining_epochs=70,corruption_level=[0.1,0.2],
                                                     pretrain_lr=0.01, training_epochs=1000, batch_size=10,hidden_layers_sizes=[n_hiddenL1,n_hiddenL2])
                    validSetPreLL = predict(validSetLL)
                    testSetPreLL = predict(testSetLL)                         
                            
                    for n_hiddenH1 in [8,7,6]:
                                for n_hiddenH2 in [5,4,3]:
                                            test_SdA(newSetHH,validSetHH,testSetHH,newLabelHH,validLabelHH,testLabelHH,
                                                     finetune_lr=0.1, pretraining_epochs=70,corruption_level=[0.1,0.2],
                                                     pretrain_lr=0.01, training_epochs=1000, batch_size=10,hidden_layers_sizes=[n_hiddenH1,n_hiddenH2])
                                            validSetPreHH = predict(validSetHH)
                                            testSetPreHH = predict(testSetHH)
        
                                            
                        
                                           

                        
                                            
                        
                                        

                                            testPreResultMat = makeResultMat(testSetPreLL,testSetPreHH,testLabelCC)
                                            validPreResultMat = makeResultMat(validSetPreLL,validSetPreHH,validLabelCC)
                        
                                            testPreList = calNum(testPreResultMat)
                                            validPreList = calNum(validPreResultMat)
                        
                                            numRight = calRight(validPreList,testPreList)
                                            RightRate = float(numRight)/len(testLabelCC)
                         
                                            print "k1 %d, k3 %d, [n_hiddenL1 %d,n_hiddenL2 %d],[n_hiddenH1 %d,n_hiddenH2 %d]  ,RightRate%f"%(k1,k3, n_hiddenL1, n_hiddenL2,n_hiddenH1,n_hiddenH2,RightRate)
                                            if (RightRate>=bestRightRate):
                                                bestRightRate = RightRate
                                                print "k1 %d, k3 %d ,[n_hiddenL1 %d,n_hiddenL2 %d],[n_hiddenH1 %d,n_hiddenH2 %d] ,bestRightRate%f"%(k1,k3, n_hiddenL1, n_hiddenL2,n_hiddenH1,n_hiddenH2,bestRightRate)
                                                Results.append([k1,k3,[n_hiddenL1,n_hiddenL2],[n_hiddenH1,n_hiddenH2],bestRightRate])
    print Results
