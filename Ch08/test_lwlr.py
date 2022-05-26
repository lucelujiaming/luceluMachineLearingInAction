# coding: utf-8

import sys
import regression
from numpy import *

xArr, yArr = regression.loadDataSet('ex0.txt')

yHat0_1 = regression.lwlr(xArr[0], xArr, yArr, 1.0)
# print("yHat0_1 : ", yHat0_1)
yHat0_003 = regression.lwlr(xArr[0], xArr, yArr, 0.0003)
# print("yHat0_003 : ", yHat0_003)

yHat = regression.lwlrTest(xArr, xArr, yArr, 0.003)
print("yHat.T : ", yHat.T)

xMat = mat(xArr)
srtInd = xMat[:,1].argsort(0)
print("yHat[srtInd] : ", yHat[srtInd])
xSort = xMat[srtInd][:,0,:]


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:,1], yHat[srtInd][:,0])
ax.scatter(xMat[:,1].flatten().A[0], \
	mat(yArr).T[:,0].flatten().A[0], s = 2, c = 'red')
plt.show()


#       # weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
#       diffTimesMat = diffMat*diffMat.T/(-2.0*k**2)
#       diffExpMat = exp(diffTimesMat)
#       # print("diffExpMat : ", diffExpMat)
#       weights[j,j] = diffExpMat[0, 0]
