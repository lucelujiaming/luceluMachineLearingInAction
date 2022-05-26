# coding: utf-8
'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
'''
from numpy import *
import operator
from os import listdir

# 有4个输入参数：
#       用于分类的输入向量是inX，
#       输入的训练样本集为dataSet，
#       标签向量为labels，
#       最后的参数k表示用于选择最近邻居的数目，其中标签向量的元素数目和矩阵dataSet的行数相同。
def classify0(inX, dataSet, labels, k):
    # 使用欧氏距离公式，计算两个向量点xA和xB之间的距离。
    # 数组的shape属性返回一个元组（tuple），元组中的元素即为NumPy数组每一个维度上的大小。
    # 获得输入的训练样本集的维度大小。
    dataSetSize = dataSet.shape[0]
    # 求输入向量和训练样本集的差。
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    # 求差的平方
    sqDiffMat = diffMat**2
    # 求和
    sqDistances = sqDiffMat.sum(axis=1)
    # 求平方和
    distances = sqDistances**0.5
    # 将x中的元素从小到大排列,提取其对应的index(索引),然后输出。
    sortedDistIndicies = distances.argsort()   
    # print("sortedDistIndicies : ", sortedDistIndicies)  
    classCount={} 
    # 然后，确定前k个距离最小元素所在的主要分类 ，输入k总是正整数；
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # print("voteIlabel : ", labels[sortedDistIndicies[i]]) 
        # 逐个给出类计数。
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    print("classCount : ", classCount) 
    # 最后，将classCount字典分解为元组列表，对元组进行排序。
    # 注意：此处的排序为逆序，即按照从最大到最小次序排序。
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回发生频率最高的元素标签。
    return sortedClassCount[0][0]

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def file2matrix(filename):
    fr = open(filename)
    # 得到文件行数
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    # 创建返回的NumPy矩阵
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    # 解析文件数据到列表
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector
    
def autoNorm(dataSet):
    # 将每列的最小值放在变量minVals中。
    minVals = dataSet.min(0)
    # 将最大值放在变量maxVals中。
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # 为了归一化特征值，我们必须使用当前值减去最小值，然后除以取值范围。
    # 们使用NumPy库中tile()函数将变量内容复制成输入矩阵同样大小的矩阵
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

# 使用分类器针对约会网站的测试代码。
def datingClassTest():
    hoRatio = 0.50      #hold out 10%
    # 首先从文件中读取数据并将其转换为归一化特征值。
    # 这个数据前三列是3种数据特征。最后一列为妹子做的标签。
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    # 接着把数据转换为归一化特征值。
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 接着计算测试向量的数量，
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    # 将这两部分数据输入到原始kNN分类器函数classify0。
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    # 最后，函数计算错误率并输出结果。
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    # print(errorCount)

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input("Percentage of games"))
    ffMiles = float(raw_input("Filer miles"))
    iceCreams = float(raw_input("iceCreams"))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt') 
    normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
    inArr = array([percentTats, ffMiles, iceCreams])
    classifierResult = classify0((inArr - minVals)/ranges, \
        normMat, datingLabels, 3)
    print("You will probly like the person: ", \
        resultList[classifierResult - 1])
        
# 这个文件的格式是这样的。每一个文件的文件名格式为：
#    <label>_<index>.txt.
# 例如文件名9_178.txt表明第178个训练文件中保存的是数字9的图像。
# 文件内容是32行32列的。表示方法一看便知。
# 这个文件就是用来读出文件内容。作为一个向量返回。
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

# 使用k-近邻算法识别手写数字的测试代码：
# 这个程序的原理是这样的。就是如果一张训练图和一张测试图画的是类似的数字。
# 那就意味着这两张图做减法求差的时候，和剩下来的1和-1比较少。
# 之后做差的平方和。算出来的数字就比较小。
# 只要找到和测试图画计算结果相比最小的那张训练图，
# 就可以认为测试图中绘制就是这张训练图代表的数字。
def handwritingClassTest():
    hwLabels = []
    # 将trainingDigits目录中的文件内容存储在列表中。
    # 注意：这个的是已经标注好的数据，不需要执行k-近邻算法。
    trainingFileList = listdir('trainingDigits')           #load the training set
    # 得到目录中有多少文件，并将其存储在变量m中。
    m = len(trainingFileList)
    # 创建一个m行1024列的训练矩阵，该矩阵的每行数据存储一个图像。
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        # 将分类标签存储在hwLabels向量中。
        hwLabels.append(classNumStr)
        # 用前面讨论的img2vector函数载入图像。
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    # 将testDigits目录中的文件内容存储在列表中。
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        # 用前面讨论的img2vector函数载入图像。
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        # 使用classify0()函数测试该目录下的每个文件。
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): 
            errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))
