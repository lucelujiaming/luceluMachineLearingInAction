# coding: utf-8

import kNN
import matplotlib
from matplotlib import pyplot as plt

# 支持中文
plt.rcParams['font.family'] = ['Songti SC']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

group, labels = kNN.createDataSet()
print("group : ", group)
print("labels : ", labels)
classData = kNN.classify0([7, 7], group, labels, 3)

print("classData : ", classData)

datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
print("datingDataMat : ", datingDataMat)
print("datingLabel : ", datingLabels)

#	fig = plt.figure()
#	ax = fig.add_subplot(111)
#	# ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
#	from numpy import *
#	ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 
#		15.0 * array(datingLabels), 15.0 * array(datingLabels))
#	plt.show()

normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
print("normMat : ", normMat)
print("ranges : ", ranges)
print("minVals : ", minVals)

