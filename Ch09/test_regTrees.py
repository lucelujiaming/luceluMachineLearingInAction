# coding: utf-8
import regTrees
from numpy import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

testMat = mat(eye(4))
print("testMat : ", testMat)

mat0,mat1 = regTrees.binSplitDataSet(testMat, 1, 0.5)
print("mat0 : \n", mat0)
print("mat1 : \n", mat1)

myDat = regTrees.loadDataSet('ex00.txt')
myMat = mat(myDat)
retTree = regTrees.createTree(myMat)
print("retTree : \n", retTree)


myDat1 = regTrees.loadDataSet('ex0.txt')
myMat1 = mat(myDat1)
retTree1 = regTrees.createTree(myMat1)
print("retTree1 : \n", retTree1)

myDat2 = regTrees.loadDataSet('ex2.txt')
myMat2 = mat(myDat2)
retTree2 = regTrees.createTree(myMat2)
print("retTree2 : \n", retTree2)

retTree2000 = regTrees.createTree(myMat2, ops= (1000, 2))
print("retTree20000 : \n", retTree2000)

myDatTest = regTrees.loadDataSet('ex2Test.txt')
myMat2test = mat(myDatTest)
retTree2Merge = regTrees.prune(retTree2, myMat2test)
print("retTree2Merge : \n", retTree2Merge)





