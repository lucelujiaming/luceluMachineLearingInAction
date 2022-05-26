# coding: utf-8

import adaboost
from numpy import *

dataMat, classLabels = adaboost.loadSimpData()
D = mat(ones((5,1))/5)
print("D : ", D)
bestStump, minError, bestClasEst = adaboost.buildStump(dataMat, classLabels, D)
print("bestStump   : ", bestStump)
print("minError    : ", minError)
print("bestClasEst : ", bestClasEst)

classifierArray, aggClassEst = adaboost.adaBoostTrainDS(dataMat, classLabels, 9)
print("classifierArray   : ", classifierArray)
print("aggClassEst   : ", aggClassEst)
classVerify = adaboost.adaClassify([0, 0], classifierArray)
print("classVerify00   : ", classVerify)

classVerify = adaboost.adaClassify([[5 ,5], [0, 0]], classifierArray)
print("classVerify   : ", classVerify)
print("classVerify50   : ", classVerify)

