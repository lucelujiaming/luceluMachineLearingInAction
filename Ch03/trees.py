# coding: utf-8
'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    # 两个特征的标签。
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

# 注意到这里计算的是最后一列的香农熵。也就是标签的香农熵。
def calcShannonEnt(dataSet):
    # 计算数据集中实例的总数
    numEntries = len(dataSet)
    # 创建一个数据字典，
    labelCounts = {}
    # the the number of unique elements and their occurance
    for featVec in dataSet: 
        # 它的键值是最后一列的数值
        currentLabel = featVec[-1]
        # 如果当前键值不存在，则扩展字典并将当前键值加入字典。
        if currentLabel not in labelCounts.keys(): 
            labelCounts[currentLabel] = 0
        # 每个键值都记录了当前类别出现的次数。
        labelCounts[currentLabel] += 1
    print("labelCounts : ", labelCounts)
    shannonEnt = 0.0
    # 使用所有类标签的发生频率计算类别出现的概率。我们将用这个概率计算香农熵。
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        # print("prob : ", prob)
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt

# 三个输入参数：待划分的数据集、划分数据集的特征、特征的返回值。
# 我们可以这样理解这段代码：
#     当我们按照某个特征划分数据集时，就需要将所有符合要求的元素抽取出来。
# 比方说，对于数据：
#      [[1, 1, 'maybe'], [1, 1, 'yes'], [1, 0, 'no'], 
#       [0, 1, 'no'],    [0, 1, 'no']]
# trees.splitDataSet(myData, 0 , 1)表明找出来以0号轴为1的元素。
# 这里所谓的0号轴，也就是上面每一个元素的第0个子元素。
# 可以看出来上面的5个元素中，第0个元素为1的是前三个。
# 也就是[1, 1, 'maybe'], [1, 1, 'yes'], [1, 0, 'no']
# 因此上，splitDataSet(myData, 0 , 1)返回的就是这三个元素。
# 同时去掉了用于分类的轴。也就是这三个元素的第0个子元素。
# 也就是：
#      [[1, 'maybe'], [1, 'yes'], [0, 'no']]
# 同样的是，对于trees.splitDataSet(myData, 0 , 0)来说，
# 上面的5个元素中，第0个元素为0的是后两个。
# 同时去掉了用于分类的轴。也就是这三个元素的第0个子元素。
# 也就是：
#      [[1, 'no'], [1, 'no']]
def splitDataSet(dataSet, axis, value):
    # 创建一个新的列表对象。
    # 这个列表保存的是一个个reducedFeatVec列表。
    retDataSet = []
    # 遍历数据集中的每个元素，
    for featVec in dataSet:
        # 一旦发现符合要求的值，
        if featVec[axis] == value:
            # 则将其添加到新创建的列表中。
            # Chop out axis used for splitting
            reducedFeatVec = featVec[:axis] 
            # 去掉用于分类的轴。
            reducedFeatVec.extend(featVec[axis+1:])
            print("reducedFeatVec : ", reducedFeatVec)
            retDataSet.append(reducedFeatVec)
            print("retDataSet : ", retDataSet)
    return retDataSet

# 接下来我们将遍历整个数据集，循环计算香农熵和splitDataSet()函数，
# 找到最好的特征划分方式。
# 该函数实现选取特征，划分数据集，计算得出最好的划分数据集的特征。
# 在函数中调用的数据需要满足一定的要求：
#     第一个要求是，数据必须是一种由列表元素组成的列表，
#                 而且所有的列表元素都要具有相同的数据长度；
#     第二个要求是，数据的最后一列或者每个实例的最后一个元素是当前实例的类别标签。
# 这个函数的原理是这样的。
#    1. 首先计算整体数据集的标签的香农熵。
#    2. 根据每一列对于数据进行划分，计算划分后的每一个类别的标签的香农熵。之后求和。
#    3. 根据香农原理。第二步得到的香农熵比第一步的香农熵小。计算差值。
#    4. 最大的差值对应的列，就是最好特征划分的索引值。
def chooseBestFeatureToSplit(dataSet):
    # 计算一行数据有几个特征。因
    # 为每一行数据除了最后一个元素是标签，剩下全是特征。
    # 因此上，特征个数等于行元素个数减一。
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    # 计算了整个数据集的原始香农熵。
    # 也就是数据集最后一列，标签列的香农熵。
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    # 循环遍历数据集中的所有特征。
    for i in range(numFeatures):        #iterate over all the features
        # 使用列表推导，将数据集中所有第i个特征值或者所有可能存在的值写入这个新list中。
        # create a list of all the examples of this feature
        # 获得数据的一列。
        featList = [example[i] for example in dataSet]
        # 通过从列表创建集合的方式去重。
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        # 遍历当前特征中的所有唯一属性值，对每个特征划分一次数据集。
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            # 计算数据集的新熵值，并对所有唯一特征值得到的熵求和。
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        # 比较所有特征中的信息增益，
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    # 返回最好特征划分的索引值
    return bestFeature                      #returns an integer

# 采用多数表决的方法决定该叶子节点的分类。
def majorityCnt(classList):
    # 使用分类名称的列表，
    classCount={}
    # 创建键值为classList中唯一值的数据字典，
    for vote in classList:
        if vote not in classCount.keys(): 
            classCount[vote] = 0
        # 字典对象存储了classList中每个类标签出现的频率，
        classCount[vote] += 1
    # 利用operator操作键值排序字典，
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回出现次数最多的分类名称。
    return sortedClassCount[0][0]

# 两个输入参数：数据集和标签列表。
def createTree(dataSet,labels):
    # 创建了名为classList的列表变量，其中包含了数据集的所有类标签。
    classList = [example[-1] for example in dataSet]
    # 第一个停止条件：所有的类标签完全相同，则直接返回该类标签。
    if classList.count(classList[0]) == len(classList): 
        return classList[0] # stop splitting when all of the classes are equal
    # 第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组
    # stop splitting when there are no more features in dataSet
    if len(dataSet[0]) == 1: 
        # 由于无法简单地返回唯一的类标签，挑选出现次数最多的类别作为返回值。
        return majorityCnt(classList)
    # 下一步开始创建树。得到最好特征划分的索引值。
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 根据索引值得到最好特征划分。
    bestFeatLabel = labels[bestFeat]
    # 字典变量myTree存储了树的所有信息，
    # 创建一个字典变量，Key是上面得到的最好特征划分。value是一棵树。
    myTree = {bestFeatLabel:{}}
    # 删除得到的最好特征划分。
    del(labels[bestFeat])
    # 创建唯一的分类标签列表。
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    # 遍历当前选择特征包含的所有属性值，
    for value in uniqueVals:
        # 复制类标签，并将其存储在新列表变量subLabels中。
        # copy all of labels, so trees don't mess up existing labels
        subLabels = labels[:]   
        # 首先根据特征和特征值调用splitDataSet把符合要求的元素挑出来。
        # 在挑出来的每个数据集划分上递归调用函数createTree()，
        # 得到的返回值将被插入到字典变量myTree中
        myTree[bestFeatLabel][value] = \
            createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            
    
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    # 使用index方法查找当前列表中第一个匹配firstStr变量的元素。
    featIndex = featLabels.index(firstStr)
    # 得到testVec变量中的key。
    key = testVec[featIndex]
    # 得到树节点的值。
    valueOfFeat = secondDict[key]
    # 代码递归遍历整棵树
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: 
        classLabel = valueOfFeat
    # 如果到达叶子节点，则返回当前节点的分类标签。
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    
