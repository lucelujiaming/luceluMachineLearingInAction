# coding: utf-8

import sys
import adaboost
from numpy import *

numIt = int(sys.argv[1])

dataArr, labelArr = adaboost.loadDataSet('horseColicTraining2.txt')
#	print("dataArr   : ", dataArr)
#	print("labelArr   : ", labelArr)

#	m = shape(dataArr)[0]
#	# 向量D非常重要，它包含了每个数据点的权重。
#	# D是一个概率分布向量，一开始的所有元素都会被初始化成1/m。
#	D = mat(ones((m,1))/m)   #init D to all equal
#	# print("D.T : ", D.T)
#	bestStump, minError, bestClasEst = adaboost.buildStump(dataArr, labelArr, D)
#	print("bestStump   : ", bestStump)
#	print("minError    : ", minError)
#	print("bestClasEst.T : ", bestClasEst.T)

classifierArray, aggClassEst = adaboost.adaBoostTrainDS(dataArr, labelArr, numIt) # 10)
print("classifierArray   : ", classifierArray)
# print("aggClassEst.T   : ", aggClassEst.T)

testArr, testLabelArr = adaboost.loadDataSet('horseColicTest2.txt')
prediction10 = adaboost.adaClassify(testArr, classifierArray)
print("prediction10.T   : ", prediction10.T)
errArr = mat(ones((67, 1)))
countErr = errArr[prediction10 != mat(testLabelArr).T]
print("countErr.sum()  : ", countErr.sum())

adaboost.plotROC(aggClassEst.T, labelArr)
