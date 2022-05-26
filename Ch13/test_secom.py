# coding: utf-8
import pca
from numpy import *

dataMat = pca.replaceNanWithMean()
meanVals = mean(dataMat, axis = 0)
meanRemoved = dataMat - meanVals
covMat = cov(meanRemoved ,rowvar = 0)
eigVals, eigVects = linalg.eig(mat(covMat))
print("eigVals : ", eigVals)

