# coding: utf-8

import logRegres

dataArr, labelMat = logRegres.loadDataSet()
print("dataArr : ", dataArr)
print("labelMat : ", labelMat)

matWeights = logRegres.gradAscent(dataArr, labelMat)
print("matRet : ", matWeights)

from numpy import *
dataMatrix = mat(dataArr) 
print("dataMatrix : ", dataMatrix)
dataMat = dataMatrix*matWeights
print("dataMat.T : ", dataMat.transpose())
sigmoidMat = logRegres.sigmoid(dataMatrix*weights) 
print("sigmoidMat : ", sigmoidMat)
print("labelMat : ", labelMat)

logRegres.plotBestFit(matWeights.getA())
