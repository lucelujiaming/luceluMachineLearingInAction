# coding: utf-8
import regTrees
from numpy import *

myDat2 = regTrees.loadDataSet('exp2.txt')
myMat2 = mat(myDat2)
retTree2 = regTrees.createTree(myMat2)
print("retTree2 : \n", retTree2)

retModeTree = regTrees.createTree(myMat2, regTrees.modelLeaf, \
	regTrees.modelErr, ops= (1, 10))
print("retModeTree : \n", retModeTree)




