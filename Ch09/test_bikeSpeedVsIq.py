# coding: utf-8
import regTrees
from numpy import *

# 数据文件中是骑自行车的速度和人的智商之间的关系。
# 分为两列。第一列是速度。第二列是智商。可以把这两列数据看成是平面上的XY坐标。
# 我们调用createTree构建回归树，对这些点进行归堆，也就是分类。
trainMat = mat(regTrees.loadDataSet('bikeSpeedVsIq_train.txt'))
testMat = mat(regTrees.loadDataSet('bikeSpeedVsIq_test.txt'))

# 使用平方误差构建回归树。
myTree = regTrees.createTree(trainMat, ops = (1 ,20))
# print("myTree : ", myTree)
# print("testMat.T : ", testMat.T)
# print("testMat[:, 0].T : ", testMat[:, 0].T)
# 传入构建出来的回归树和测试数据的第一列表示的车速进行预测。
yHat = regTrees.createForeCast(myTree, testMat[:, 0])
# print("yHat.T : ", yHat.T)
# 通过查看使用平方误差构建出来的回归树得到的皮尔逊积矩相关系数，
# 得到预测出来人的智商和测试集中的人的智商对比。
corrcoefRet = corrcoef(yHat, testMat[:, 1], rowvar = 0)
print("corrcoefRet : ", corrcoefRet)

"""
# 使用模型树。
myModeTree = regTrees.createTree(trainMat, regTrees.modelLeaf, \
	regTrees.modelErr, ops = (1 ,20))
yModeHat = regTrees.createForeCast(myModeTree, testMat[:, 0],
	regTrees.modelTreeEval)
# print("yModeHat.T : ", yModeHat.T)
# 查看使用模型树构建出来的回归树得到的皮尔逊积矩相关系数。
corrcoefModeRet = corrcoef(yModeHat, testMat[:, 1], rowvar = 0)
print("corrcoefModeRet : ", corrcoefModeRet)

# 使用线性回归。
ws, X, Y = regTrees.linearSolve(trainMat)
m=len(testMat)
ylinearHat = mat(zeros((m,1)))
print("ws : ", ws)
for i in range(shape(testMat)[0]):
	ylinearHat[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
# 查看使用线性回归构建出来的回归树得到的皮尔逊积矩相关系数。
corrcoefLinearRet = corrcoef(ylinearHat, testMat[:, 1], rowvar = 0)
print("corrcoefLinearRet : ", corrcoefLinearRet)
"""