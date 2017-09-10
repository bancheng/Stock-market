# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:50:50 2017

@author: Administrator
"""

from featureConstruction import featureConstruction

from littleFun import calNum,calRight

#import theano.tensor as T
import numpy as np 
import operator


from DataPreparation import finalPre,splitData,loadData
from littleFun import autoNorm
from SVMRF import SVM
   
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
    
def makeResultMat(PreLL,PreHH,PreCC):
    ResultMat = np.zeros((PreCC.shape[0],3))
    for i in range(PreCC.shape[0]):
        ResultMat[i][0] = PreLL[i]
        ResultMat[i][1] = PreHH[i]
        ResultMat[i][2] = PreCC[i]
    return ResultMat
    
# 用CC 为标签进行分类
if __name__ == '__main__':
    f = open("SVMCC.txt","w+")
    print "直接以收盘价为标签进行预测："
    f.write("直接以收盘价为标签进行预测："+"\n")
  

    dataSets,dataLabelCC = dataPreCC()
     #划分数据集

    trainSetCC,validSetCC,testSetCC,trainLabelCC,validLabelCC,testLabelCC = splitData(dataSets,dataLabelCC,trainRate=0.9,validRate=0.05,testRate=0.05)
    # 将Label的集合换成arraylike，进而使用SVM函数库
 
    
    trainLabelCC = np.array(trainLabelCC)
    validLabelCC = np.array(validLabelCC)
    testLabelCC = np.array(testLabelCC)
    #初始化模型
    

    bestRightRate = 0.0
    Results=[]


    for C in [0.1,0.5,1.0,5.0,10.0]:
        clfCC = SVM(C=C)
        clfCC.fit(trainSetCC,trainLabelCC)
        RightRate = clfCC.score(testSetCC,testLabelCC)

                         
        print "C %f,RightRate%f"%(C,RightRate)
        f.write(" C%f,RightRate%f"%(C,RightRate) + "\n")
        if (RightRate>=bestRightRate):
            bestRightRate = RightRate
            f.write(" C %f ,bestRightRate%f"%(C,bestRightRate)+"\n")
            print " C%f ,bestRightRate%f"%(C,bestRightRate)
            Results.append([C,bestRightRate])
    print Results
    for i in range(len(Results)):
        f.write("Best Parmaters:")
        f.write(" C %f ,bestRightRate%f"%(Results[i][0],Results[i][1]))
    f.close()


#不经过KNN筛选
if __name__ == '__main__':
    f = open("SVMLH.txt","w+")
    print"不经过KNN筛选，以最低价及最高价对收盘价进行预测："
    f.write("不经过KNN筛选，以最低价及最高价对收盘价进行预测："+"\n")
    trainSetLL,validSetLL,testSetLL,trainLabelLL,validLabelLL,testLabelLL,trainSetHH,validSetHH,testSetHH,trainLabelHH,validLabelHH,testLabelHH,trainSetCC,validSetCC,testSetCC,trainLabelCC,validLabelCC,testLabelCC = finalPre()
    # 将Label的集合换成arraylike，进而使用SVM函数库
    trainLabelLL = np.array(trainLabelLL)
    validLabelLL = np.array(validLabelLL)
    testLabelLL = np.array(testLabelLL)
    
    trainLabelHH = np.array(trainLabelHH)
    validLabelHH = np.array(validLabelHH)
    testLabelHH = np.array(testLabelHH)
    
    trainLabelCC = np.array(trainLabelCC)
    validLabelCC = np.array(validLabelCC)
    testLabelCC = np.array(testLabelCC)
    #初始化模型

    bestRightRate = 0.0
    Results=[]


    for C1 in [0.1,0.5,1.0,5.0,10.0]:
        clfLL = SVM(C=C1)
        clfLL.fit(trainSetLL,trainLabelLL)
        validSetPreLL = clfLL.predict(validSetLL)
        testSetPreLL = clfLL.predict(testSetLL)
        for C2 in [0.1,0.5,1.0,5.0,10.0]:
            
            clfHH = SVM(C=C2)
            
            clfHH.fit(trainSetHH,trainLabelHH)
    
           
            validSetPreHH = clfHH.predict(validSetHH)
            testSetPreHH = clfHH.predict(testSetHH)
                            
            testPreResultMat = makeResultMat(testSetPreLL,testSetPreHH,testLabelCC)
            validPreResultMat = makeResultMat(validSetPreLL,validSetPreHH,validLabelCC)
    
            testPreList = calNum(testPreResultMat)
            validPreList = calNum(validPreResultMat)
                        
            numRight = calRight(validPreList,testPreList)
            RightRate = float(numRight)/len(testLabelCC)
                         
            print "C1% f, C2 %f, RightRate%f"%(C1,C2,RightRate)
            f.write(" C1% f, C2 %f, RightRate%f"%(C1,C2,RightRate) + "\n")
            if (RightRate>=bestRightRate):
                bestRightRate = RightRate
                f.write(" C1% f, C2 %f , bestRightRate%f"%(C1,C2,bestRightRate)+"\n")
                print " C1% f, C2 %f , bestRightRate%f"%(C1,C2,bestRightRate)
                Results.append([C1,C2,bestRightRate])
    print Results
    for i in range(len(Results)):
        f.write("Best Parmaters:")
        f.write(" C1% f, C2 %f , bestRightRate%f"%(Results[i][0],Results[i][1],Results[i][2]))
    f.close()


    
    
# 经过KNN 筛选    
if __name__ == '__main__':
    f = open("SVMKNNLH.txt","w+")
    print"经过KNN筛选，以最低价及最高价对收盘价进行预测："
    f.write("经过KNN筛选，以最低价及最高价对收盘价进行预测：")
    trainSetLL,validSetLL,testSetLL,trainLabelLL,validLabelLL,testLabelLL,trainSetHH,validSetHH,testSetHH,trainLabelHH,validLabelHH,testLabelHH,trainSetCC,validSetCC,testSetCC,trainLabelCC,validLabelCC,testLabelCC = finalPre()

  # 将Label的集合换成arraylike，进而使用SVM函数库
    trainLabelLL = np.array(trainLabelLL)
    validLabelLL = np.array(validLabelLL)
    testLabelLL = np.array(testLabelLL)
    
    trainLabelHH = np.array(trainLabelHH)
    validLabelHH = np.array(validLabelHH)
    testLabelHH = np.array(testLabelHH)
    
    trainLabelCC = np.array(trainLabelCC)
    validLabelCC = np.array(validLabelCC)
    testLabelCC = np.array(testLabelCC)

    bestRightRate = 0.0
    Results=[]
 
    for k1 in [3,5,7,9]:
        rightDataLL,rightDataLabelLL,wrongDataLL,wrongDataLabelLL =  seperateData(trainSetLL,trainLabelLL,k1)
        for k2 in [3,5,7,9]:
            rightData1LL,rightDataLabel1LL,wrongData1LL,wrongDataLabel1LL =  seperateData(wrongDataLL,wrongDataLabelLL,k2)
            newSetLL = np.concatenate((rightDataLL,rightData1LL))
                        
            newLabelLL = np.concatenate((rightDataLabelLL,rightDataLabel1LL))
            for k3 in [3,5,7,9]:
                rightDataHH,rightDataLabelHH,wrongDataHH,wrongDataLabelHH =  seperateData(trainSetHH,trainLabelHH,k3)
                for k4 in [3,5,7,9]:
                    rightData1HH,rightDataLabel1HH,wrongData1HH,wrongDataLabel1HH =  seperateData(wrongDataHH,wrongDataLabelHH,k4)
                    
                    newSetHH = np.concatenate((rightDataHH,rightData1HH))
                    newLabelHH = np.concatenate((rightDataLabelHH,rightDataLabel1HH))
                    for C1 in [0.1,0.5,1.0,5.0,10.0]:
                        clfLL = SVM(C=C1)
                        clfLL.fit(newSetLL,newLabelLL)
                        validSetPreLL = clfLL.predict(validSetLL)
                        testSetPreLL = clfLL.predict(testSetLL)
                        for C2 in [0.1,0.5,1.0,5.0,10.0]:
                                #初始化模型
                            
                            clfHH = SVM(C=C2)
                           
                            
                        
                            
                            
                            
                            
                            clfHH.fit(newSetHH,newLabelHH)
    
                            
                            validSetPreHH = clfHH.predict(validSetHH)
                            testSetPreHH = clfHH.predict(testSetHH)
                            
                            testPreResultMat = makeResultMat(testSetPreLL,testSetPreHH,testLabelCC)
                            validPreResultMat = makeResultMat(validSetPreLL,validSetPreHH,validLabelCC)
    
                            testPreList = calNum(testPreResultMat)
                            validPreList = calNum(validPreResultMat)
                        
                            numRight = calRight(validPreList,testPreList)
                            RightRate = float(numRight)/len(testLabelCC)
                         
                            print "k1 %d, k2 %d, k3 %d, k4 %d C1% f, C2 %f, RightRate%f"%(k1,k2,k3,k4,C1,C2,RightRate)
                            f.write("k1%d, k2 %d, k3 %d, k4 %d C1% f, C2 %f, RightRate%f"%(k1,k2,k3,k4,C1,C2,RightRate) + "\n")
                            if (RightRate>=bestRightRate):
                                bestRightRate = RightRate
                                f.write("k1 %d, k2 %d, k3 %d, k4 %d C1% f, C2 %f , bestRightRate%f"%(k1,k2,k3,k4,C1,C2,bestRightRate)+"\n")
                                print "k1 %d, k2 %d, k3 %d, k4 %d C1% f, C2 %f , bestRightRate%f"%(k1,k2,k3,k4,C1,C2,bestRightRate)
                                Results.append([k1,k2,k3,k4,C1,C2,bestRightRate])
    print Results
    for i in range(len(Results)):
        f.write("Best Parmaters:")
        f.write(" k1 %d, k2 %d, k3 %d, k4 %d ,C1% f, C2 %f , bestRightRate %f"%(Results[i][0],Results[i][1],Results[i][2],Results[i][3],Results[i][4],Results[i][5],Results[i][6]))
        f.write("\n")
    f.close()
   
    
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    