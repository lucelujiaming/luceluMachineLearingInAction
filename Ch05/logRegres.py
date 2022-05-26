# coding: utf-8
'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''
import matplotlib
matplotlib.use('TkAgg')
from numpy import *

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    # 这个文件每行前两个值分别是X和Y，第三个值是数据对应的类别标签。
    # 例如：[1.176813  3.167020    1]表明(1.176813, 3.167020)属于类型1。
    for line in fr.readlines():
        lineArr = line.strip().split()
        # 把每一行的X和Y放入数据集。
        # 给数据集补充一个维度 x0，一般情况下x0 = 1
        # 在线性回归的公式是这样的θ0+θ1x1+θ2*x2+.... ，
        # 所以数据集前面应该加一列1，来充当x0
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        # 把第三列放入类型标签。
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

# 梯度上升法的伪代码如下：
# 每个回归系数初始化为1 
# 重复R次：
#   计算整个数据集的梯度
#   使用alpha × gradient更新回归系数的向量
#   返回回归系数
# 该函数有两个参数。
#       第一个参数是dataMathIn，它是一个2维NumPy数组，
#       每列分别代表每个不同的特征，每行则代表每个训练样本。
#       第二个参数是类别标签，它是一个1×100的行向量。
# 这个算法的逻辑是这样的。
#   我们已知输入数据dataMatIn，和输出结果classLabels。
#   我们认为这两者的关系就是：
#   classLabels = sigmoid(dataMatrix * weights)
#   因此上问题转换为求weights。
#   又因为classLabels要么为零，要么为一。
#   因此上，
def gradAscent(dataMatIn, classLabels):
    # 获得输入数据并将它们转换成NumPy矩阵。
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    # 为了便于矩阵运算，需要将该行向量转置，转换为列向量，
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix
    # 得到矩阵大小
    m,n = shape(dataMatrix)
    # 变量alpha是向目标移动的步长，
    alpha = 0.001
    # maxCycles是迭代次数。
    maxCycles = 500
    # 把一列数据变成矩阵对角线矩阵，之后乘上sigmoid
    weights = ones((n,1))
    for k in range(maxCycles):              #heavy on matrix operations
        # 注意：这里是操作整个dataMatIn，速度比较慢。
        h = sigmoid(dataMatrix*weights)     #matrix mult
        # 计算真实类别与预测类别的差值
        error = (labelMat - h)              #vector subtraction
        # 按照该差值的方向调整回归系数。
        # 详细的数学推导参见：
        #      https://zhuanlan.zhihu.com/p/28922957
        #      https://www.cnblogs.com/zhongmiaozhimen/p/6155093.html
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
    return weights

# 分析数据：画出决策边界。
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    # 加载数据。
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    # 获得数据中坐标点的个数。
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    # 根据坐标点属于的类别标签，把坐标点加到两个list中。
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    # 开始绘图。
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制两个坐标点list。
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    # 给出X / Y坐标范围。
    x = arange(-3.0, 3.0, 0.1)
    # 决策边界是: w0 + w1*x1 + w2*x2 = 0，所以x2 = (-w0-w1*x1)/w2
    # y = (- 4.12414349 - 0.48007329 * x) / -0.6168482
    #   = 0.778268 x + 6.6858320
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

# 所有回归系数初始化为1 
# 对数据集中每个样本
#   计算该样本的梯度
#   使用alpha × gradient更新回归系数值
# 返回回归系数值
def stocGradAscent0(dataMatrix, classLabels):
    # 得到矩阵大小
    m,n = shape(dataMatrix)
    # 变量alpha是向目标移动的步长，
    alpha = 0.01
    # 把一列数据变成矩阵对角线矩阵，之后乘上sigmoid
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        # 和gradAscent相比，这里是操作dataMatrix的一个元素。
        h = sigmoid(sum(dataMatrix[i]*weights))
        # 计算真实类别与预测类别的差值
        error = classLabels[i] - h
        # 按照该差值的方向调整回归系数。
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    # 得到矩阵大小
    m,n = shape(dataMatrix)
    # 把一列数据变成矩阵对角线矩阵，之后乘上sigmoid
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # 一方面，alpha在每次迭代的时候都会调整，
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not 
            # 通过随机选取样本来更新回归系数。
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            # 计算真实类别与预测类别的差值。
            error = classLabels[randIndex] - h
            # 按照该差值的方向调整回归系数。
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def classifyVector(inX, weights):
    # 以回归系数和特征向量作为输入来计算对应的Sigmoid值。
    prob = sigmoid(sum(inX*weights))
    # 如果Sigmoid值大于0.5函数返回1，否则返回0。
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    # 打开测试集和训练集
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    # 计算回归系数向量。
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0; numTestVec = 0.0
    # 在系数计算完成之后，导入测试集。
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    # 计算分类错误率。
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))
        