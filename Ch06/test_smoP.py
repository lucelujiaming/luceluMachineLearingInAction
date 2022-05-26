# coding: utf-8

import svmMLiA
from numpy import *

dataList,labelList = svmMLiA.loadDataSet('testSet.txt')
print("dataList : ", dataList)
print("labelList : ", labelList)

b, alphas = svmMLiA.smoP(dataList,labelList, 0.6, 0.001, 40)
print("b :", b)
print("alphas.T :", alphas.T)
print("alphas[alphas > 0] :", alphas[alphas > 0])
print("alphas.shape :", shape(alphas[alphas > 0]))

ws = svmMLiA.calcWs(alphas, dataList,labelList)
print("ws : ", ws)

dataMat = mat(dataList)
print("ws : ", dataMat[0] * mat(ws) + b)


print("svmMLiA.testRbf : ", svmMLiA.testRbf())
