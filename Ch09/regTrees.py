# coding: utf-8
'''
Created on Feb 4, 2011
Tree-Based Regression Methods
@author: Peter Harrington
'''
from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # Convert map to list to avoid this mistake
        # TypeError: unsupported operand type(s) for /: 'map' and 'int'
        fltLine = list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

# 该函数有3个参数：数据集合、待切分的特征和该特征的某个值。
def binSplitDataSet(dataSet, feature, value):
    # 在给定特征和特征值的情况下，
    # 该函数通过数组过滤方式将上述数据集合切分得到两个子集
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    # 原书代码报错 index 0 is out of bounds,使用上面两行代码
    # mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][0]
    # mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:][0]
    # 返回切分的子集。
    return mat0,mat1

# 当数确定不再对数据进行切分时，将调用该函数来得到叶节点的模型。
def regLeaf(dataSet):#returns the value used for each leaf
    # 该模型其实就是目标变量的均值。
    # mean()函数功能：求取均值。
    return mean(dataSet[:,-1])
# 总方差计算函数：在给定数据上计算目标变量的平方误差。
def regErr(dataSet):
    # 直接调用均方差函数var。乘以数据集中样本的个数，得到总方差。
    return var(dataSet[:,-1]) * shape(dataSet)[0]

# 下面的三个函数是模型树的相关函数。
# 核心思想就是不把叶节点简单地设定为常数值，而是把叶节点设定为分段线性函数，
# 将数据集格式化成目标变量Y和自变量X。与第8章一样，X和Y用于执行简单的线性回归。
# 另外在这个函数中也应当注意，如果矩阵的逆不存在也会造成程序异常。
def linearSolve(dataSet):   #helper function used in two places
    # 获取数据大小。
    m,n = shape(dataSet)
    #create a copy of data with 1 in 0th postion
    # 使用岭回归。
    X = mat(ones((m,n))); Y = mat(ones((m,1)))
    # 把数据集的前n - 1列作为X，最后一列作为Y。
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    # 开始执行简单的线性回归。
    xTx = X.T*X
    # 如果矩阵的逆不存在也会造成程序异常。
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    # X和Y用于执行简单的线性回归。
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet):#create linear model and return coeficients
    # 调用linearSolve函数。
    ws,X,Y = linearSolve(dataSet)
    # 返回回归系数ws。
    return ws
# 在给定的数据集上计算误差。
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    # 返回yHat和Y之间的平方误差。
    return sum(power(Y - yHat,2))

# 它是回归树构建的核心函数。该函数的目的是找到数据的最佳二元切分方式。
#   寻找方法就是遍历。
# 如果找不到一个“好”的二元切分，该函数返回None并
# 同时调用createTree()方法来产生叶节点，叶节点的值也将返回None。
# 包括三种情况：
#  1. 如果该数目为1，那么就不需要再切分而直接返回。
#  2. 如果切分数据集后效果提升不够大，那么就不应进行切分操作而直接创建叶节点。
#  1. 如果某个子集的大小小于用户定义的参数tolN，那么也不应切分。
# 伪代码大致如下：
#   对每个特征：
#       对每个特征值：
#           将数据集切分成两份
#           计算切分的误差
#           如果当前误差小于当前最小误差，那么将当前切分设定为最佳切分并更新最小误差
#   返回最佳切分的特征和阈值
# 包含四个参数：
#   分别是数据集，叶节点生成函数，误差估计函数指针。函数的停止时机参数。
#   其中变量tolS是容许的误差下降值，tolN是切分的最少样本数。
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    # 一开始为ops设定了tolS和tolN，用于控制函数的停止时机。
    # 变量tolS是容许的误差下降值，tolN是切分的最少样本数。
    tolS = ops[0]; tolN = ops[1]
    #if all the target variables are the same value: quit and return value
    # 1. 如果该数目为1，那么就不需要再切分而直接返回
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #exit cond 1
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    #the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    # 在所有可能的特征及其可能取值上遍历，找到最佳的切分方式。
    for featIndex in range(n-1):
        # for splitVal in set(dataSet[:,featIndex]):
        for splitVal in set((dataSet[:, featIndex].T.A.tolist())[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 如果某个子集的大小小于用户定义的参数tolN，那么也不应切分。
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): 
                continue
            # 最佳切分也就是使得切分后能达到最低误差的切分。
            newS = errType(mat0) + errType(mat1)
            if newS < bestS: 
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #if the decrease (S-bestS) is less than a threshold don't do the split
    # 2. 如果切分数据集后效果提升不够大，那么就不应进行切分操作而直接创建叶节点。
    if (S - bestS) < tolS: 
        return None, leafType(dataSet) #exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 如果某个子集的大小小于用户定义的参数tolN，那么也不应切分。
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #exit cond 3
        return None, leafType(dataSet)
    # 如果这些提前终止条件都不满足，那么就返回切分特征和特征值。
    return bestIndex,bestValue#returns the best feature to split on
                              #and the value used for that split

# 伪代码大致如下：
#   找到最佳的待切分特征：
#       如果该节点不能再分，将该节点存为叶节点
#       执行二元切分
#       在右子树调用createTree()方法
#       在左子树调用createTree()方法
# 树构建函数createTree有4个参数：
#    数据集和其他3个可选参数。
#       leafType给出建立叶节点的函数；
#       errType代表误差计算函数；
#       ops是一个包含树构建所需其他参数的元组。
# Assume dataSet is NumPy Mat so we can array filtering
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    # 数首先尝试调用chooseBestSplit函数将数据集分成两个部分，
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#choose the best split
    # 如果满足停止条件，chooseBestSplit()将返回None和某类模型的值
    if feat == None: 
        return val #if the splitting hit a stop condition return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    # 如果不满足停止条件，将创建一个新的Python字典并将数据集分成两份，
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    # 在这两份数据集上将分别继续递归调用
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree  

# 通过测试输入变量是否是一棵树，判断当前处理的节点是否是叶节点。
def isTree(obj):
    return (type(obj).__name__=='dict')

# 返回树平均值。
def getMean(tree):
    # 从上往下遍历树直到叶节点为止。
    if isTree(tree['right']): 
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): 
        tree['left'] = getMean(tree['left'])
    # 如果找到两个叶节点则计算它们的平均值。
    return (tree['left']+tree['right'])/2.0

# 后剪枝函数。
#   从上而下找到叶节点，用测试集来判断将这些叶节点合并
#   是否能降低测试误差。如果是的话就合并。
# 伪代码如下：
#   基于已有的树切分测试数据：
#       如果存在任一子集是一棵树，则在该子集递归剪枝过程
#       计算将当前两个叶节点合并后的误差
#       计算不合并的误差
#       如果合并会降低误差的话，就将叶节点合并
# 两个参数：待剪枝的树与剪枝所需的测试数据testData。
# 树遍历采用的是深度优先。
def prune(tree, testData):
    # 首先需要确认测试集是否为空
    if shape(testData)[0] == 0:
        return getMean(tree) #if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):
    #if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    # 一旦非空，则反复递归调用函数进行切分。
    if isTree(tree['left']): 
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): 
        tree['right'] =  prune(tree['right'], rSet)
    # if they are now both leafs, see if we can merge them
    # 如果两个分支已经不再是子树，那么就可以进行合并。
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        # 具体做法是对合并前后的误差进行比较。
        # 计算不合并的误差。
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2))
        # 计算合并的误差。
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        # 如果合并后的误差比不合并的误差小就进行合并操作，反之则不合并直接返回。
        if errorMerge < errorNoMerge: 
            print("merging")
            return treeMean
        # 反之则不合并直接返回。
        else: 
            return tree
    else: 
        return tree

# 下面的是针对测试集的测试算法的代码。
# 这个是针使用平方误差构建出来的回归树使用的测试比较函数。
def regTreeEval(model, inDat):
    return float(model)

# 这个是模型树使用的测试比较函数。
def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)
# 对于输入的单个数据点或者行向量，函数返回一个浮点值。
def treeForeCast(tree, inData, modelEval=regTreeEval):
    # 在给定树结构的情况下，对于单个数据点，该函数会给出一个预测值。
    if not isTree(tree): 
        return modelEval(tree, inData)
    # 如果不是单个数据点，就根据节点值迭代调用treeForeCast函数。
    # 比当前节点大就走左边。
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']): 
            return treeForeCast(tree['left'], inData, modelEval)
        else: 
            # 一旦到达叶节点，它就会在输入数据上调用modelEval()函数，
            # 而该函数的默认值是regTreeEval()。
            return modelEval(tree['left'], inData)
    # 比当前节点小就走右边。
    else:
        if isTree(tree['right']): 
            return treeForeCast(tree['right'], inData, modelEval)
        else: 
            # 一旦到达叶节点，它就会在输入数据上调用modelEval()函数，
            # 而该函数的默认值是regTreeEval()。
            return modelEval(tree['right'], inData)

# 函数包括三个参数：
#    构建好的回归树，测试集的第一列，测试比较函数。
# 返回根据测试集的第一列预测出来的结果。用于和测试集实际的第二列进行比较。
# 测试比较函数这份代码中有两个，分别用于
# 使用平方误差构建出来的回归树和模型树。
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = mat(zeros((m,1)))
    # 多次调用treeForeCast()函数。
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    # 以向量形式返回一组预测值
    return yHat

