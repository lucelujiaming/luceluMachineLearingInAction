# coding: utf-8

import logRegres
from numpy import *

dataArr, labelMat = logRegres.loadDataSet()
matWeights0 = logRegres.stocGradAscent0(array(dataArr), labelMat)
print("matRet : ", matWeights0)

logRegres.plotBestFit(matWeights0)

matWeights1 = logRegres.stocGradAscent1(array(dataArr), labelMat)
print("matRet : ", matWeights1)

logRegres.plotBestFit(matWeights1)


matWeights500 = logRegres.stocGradAscent1(array(dataArr), labelMat, 500)
print("matRet : ", matWeights500)

logRegres.plotBestFit(matWeights500)

