# coding: utf-8

import bayes
import matplotlib
from matplotlib import pyplot as plt

listOPosts, listClasses = bayes.loadDataSet()
print("listOPosts : ", listOPosts)
print("listClasses : ", listClasses)
# 1. 创建一个包含在所有文档中出现的不重复词的列表。作为整体数据集。
myVocalList = bayes.createVocabList(listOPosts)
print("myVocalList : ", myVocalList)
mySortedVocalList = sorted(myVocalList)
print("mySortedVocalList : ", mySortedVocalList)

#	vecFst = bayes.setOfWords2Vec(myVocalList, listOPosts[0])
#	print("vecFst : ", vecFst)
#	vecTrd = bayes.setOfWords2Vec(myVocalList, listOPosts[3])
#	print("vecTrd : ", vecTrd)

trainMat = []
for postinDoc in listOPosts:
	# 2. 使用每一个词条，针对整体数据集创建整体数据集的文档向量。
	trainMat.append(bayes.setOfWords2Vec(myVocalList, postinDoc))
print("trainMat : ", trainMat)

# 3. 传入traindNB0()函数中用于计算分类所需的概率。
p0V, p1V, pAb = bayes.trainNB0(trainMat, listClasses)
print("p0V : ", p0V)
print("p1V : ", p1V)
print("pAb : ", pAb)

bayes.testingNB()
