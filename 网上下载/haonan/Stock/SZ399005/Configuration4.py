# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 07:00:29 2017

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


#   纯粹对CC为标签进行回归（最原始的对照函数）
if __name__ == '__main__': 
    print "直接以CC为标签的结果:"
    dataSets,dataLabelCC = dataPreCC()   
    trainSetCC,validSetCC,testSetCC,trainLabelCC,validLabelCC,testLabelCC = splitData(dataSets,dataLabelCC,trainRate=0.9,validRate=0.05,testRate=0.05)

    
    for n_hiddenC1 in [8,7,6]:
        for n_hiddenC2 in [5,4,3]:
            test_SdA(trainSetCC,validSetCC,testSetCC,trainLabelCC,validLabelCC,testLabelCC,
                                                     finetune_lr=0.1, pretraining_epochs=70,corruption_level=[0.1,0.2],
                                                     pretrain_lr=0.01, training_epochs=1000, batch_size=10,hidden_layers_sizes=[n_hiddenC1,n_hiddenC2])
            validSetPreCC = predict(validSetCC)
            testSetPreCC = predict(testSetCC)
            Error = 0
            for  i in range(len(testSetPreCC)):
                if(testSetPreCC[i]!=testLabelCC[i]):
                    Error+=1
            print "n_hiddenC1 %d, n_hiddenC2 %d , rightRate %f "%(n_hiddenC1,n_hiddenC2 , (1.0 - float(Error)/len(testLabelCC)))
# 对数据不进行KNN分类的实验结果（对比实验）    
''' 
'''   
if __name__ == '__main__':
    print "对数据不进行KNN分类以LL,HH判定CC得到的结果: " 
    trainSetLL,validSetLL,testSetLL,trainLabelLL,validLabelLL,testLabelLL,trainSetHH,validSetHH,testSetHH,trainLabelHH,validLabelHH,testLabelHH,trainSetCC,validSetCC,testSetCC,trainLabelCC,validLabelCC,testLabelCC = finalPre()

    bestRightRate = 0.0
    Results = []

    for n_hiddenL1 in [8,7,6]:
        for n_hiddenL2 in [5,4,3]:
            for n_hiddenH1 in [8,7,6]:
                for n_hiddenH2 in [5,4,3]:
                    test_SdA(trainSetLL,validSetLL,testSetLL,trainLabelLL,validLabelLL,testLabelLL,
                                                     finetune_lr=0.1, pretraining_epochs=70,corruption_level=[0.1,0.2],
                                                     pretrain_lr=0.01, training_epochs=1000, batch_size=10,hidden_layers_sizes=[n_hiddenL1,n_hiddenL2])
                    
                    validSetPreLL = predict(validSetLL)
                    testSetPreLL = predict(testSetLL)
                        
                    test_SdA(trainSetHH,validSetHH,testSetHH,trainLabelHH,validLabelHH,testLabelHH,
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
                         
                    print "hiddenL1 %d,hiddenL2 %d, hiddenH1 %d, hiddenH2 %d, RightRate%f"%(n_hiddenL1, n_hiddenL2, n_hiddenH1,n_hiddenH2, RightRate)
                    if (RightRate>=bestRightRate):
                        bestRightRate = RightRate
                        print "hiddenL1 %d,hiddenL2 %d, hiddenH1 %d, hiddenH2 %d ,bestRightRate%f"%( n_hiddenL1, n_hiddenL2, n_hiddenH1,n_hiddenH2,bestRightRate)
                        Results.append([n_hiddenL1, n_hiddenL2, n_hiddenH1,n_hiddenH2,bestRightRate])
    print Results
                      

    
if __name__ == '__main__':
    print "对数据进行KNN处理后以LL,HH决定CC得到的实验结果:"
    trainSetLL,validSetLL,testSetLL,trainLabelLL,validLabelLL,testLabelLL,trainSetHH,validSetHH,testSetHH,trainLabelHH,validLabelHH,testLabelHH,trainSetCC,validSetCC,testSetCC,trainLabelCC,validLabelCC,testLabelCC = finalPre()

   
    bestRightRate = 0.0
    Results = []
    for k1 in  [3,5,7,9]:
        rightDataLL,rightDataLabelLL,wrongDataLL,wrongDataLabelLL =  seperateData(trainSetLL,trainLabelLL,k1)
        for k2 in [3,5,7,9]:
            rightData1LL,rightDataLabel1LL,wrongData1LL,wrongDataLabel1LL =  seperateData(wrongDataLL,wrongDataLabelLL,k2)
            newSetLL = np.concatenate((rightDataLL,rightData1LL))
            newLabelLL = np.concatenate((rightDataLabelLL,rightDataLabel1LL))
            for k3 in  [3,5,7,9]:
                rightDataHH,rightDataLabelHH,wrongDataHH,wrongDataLabelHH =  seperateData(trainSetHH,trainLabelHH,k3)
                
                for k4 in  [3,5,7,9]:
                    rightData1HH,rightDataLabel1HH,wrongData1HH,wrongDataLabel1HH =  seperateData(wrongDataHH,wrongDataLabelHH,k4)
                    
                        
                    

                        
                    newSetHH = np.concatenate((rightDataHH,rightData1HH))
                    newLabelHH = np.concatenate((rightDataLabelHH,rightDataLabel1HH))
                    
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
                         
                                            print "k1 %d, k2 %d, k3 %d, k4 %d [n_hiddenL1 %d,n_hiddenL2 %d],[n_hiddenH1 %d,n_hiddenH2 %d]  ,RightRate%f"%(k1,k2,k3,k4, n_hiddenL1, n_hiddenL2,n_hiddenH1,n_hiddenH2,RightRate)
                                            if (RightRate>=bestRightRate):
                                                bestRightRate = RightRate
                                                print "k1 %d, k2 %d, k3 %d, k4 %d ,[n_hiddenL1 %d,n_hiddenL2 %d],[n_hiddenH1 %d,n_hiddenH2 %d] ,bestRightRate%f"%(k1,k2,k3,k4, n_hiddenL1, n_hiddenL2,n_hiddenH1,n_hiddenH2,bestRightRate)
                                                Results.append([k1,k2,k3,k4,[n_hiddenL1,n_hiddenL2],[n_hiddenH1,n_hiddenH2],bestRightRate])
    print Results
