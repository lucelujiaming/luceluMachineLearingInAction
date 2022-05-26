# coding: utf-8

import trees
import matplotlib
from matplotlib import pyplot as plt

myData, labels = trees.createDataSet()
print("myData : ", myData)
print("labels : ", labels)

valShannonEnt = trees.calcShannonEnt(myData)
print("valShannonEnt : ", valShannonEnt)

myData[0][-1] = 'maybe'
print("myData : ", myData)
valNewShannonEnt = trees.calcShannonEnt(myData)
print("valNewShannonEnt : ", valNewShannonEnt)

valsplitDataSet = trees.splitDataSet(myData, 0 , 1)
# 
print("valsplitDataSet : ", valsplitDataSet)

valsplitDataSet = trees.splitDataSet(myData, 0 , 0)
print("valsplitDataSet : ", valsplitDataSet)

bestIndex = trees.chooseBestFeatureToSplit(myData)
print("bestIndex : ", bestIndex)

myTree = trees.createTree(myData, labels)
print("myTree : ", myTree)
