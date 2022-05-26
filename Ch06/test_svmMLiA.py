# coding: utf-8

import svmMLiA
from numpy import *

dataList,labelList = svmMLiA.loadDataSet('testSet.txt')
print("dataList : ", dataList)
print("labelList : ", labelList)

b, alphas = svmMLiA.smoSimple(dataList,labelList, 0.6, 0.001, 40)
print("b :", b)
print("alphas.T :", alphas.T)
print("alphas.T :", alphas[alphas > 0])
print("alphas.T :", shape(alphas[alphas > 0]))
for i in range(100):
	if alphas[i] > 0.0:
		print("dataList[", i,"] = ", dataList[i], 
			  "labelList[", i,"] = ", labelList[i])

