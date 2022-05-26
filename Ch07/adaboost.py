# coding: utf-8
'''
Created on Nov 28, 2010
Adaboost is short for Adaptive Boosting
@author: Peter
'''
from numpy import *

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

# #单层决策树的阈值过滤函数
# 第一个函数stumpClassify()是通过阈值比较对数据进行分类的。
# 所有在阈值一边的数据会分到类别-1，而在另外一边的数据分到类别+1。
# 输入参数： 输⼊待分类的数据， 输⼊数据的某个特征。
#           设定的阈值，阈值⽐较。
# 返回分类的结果。
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    # 对数据集每一列的各个特征进行阈值过滤
    # 该函数可以通过数组过滤来实现，首先将返回数组的全部元素设置为1，
    retArray = ones((shape(dataMatrix)[0],1))
    # 然后将所有不满足不等式要求的元素设置为-1。
    # 可以基于数据集中的任一元素进行比较，同时也可以将不等号在大于、小于之间切换
    # 阈值的模式，将小于某一阈值的特征归类为-1
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    # 将大于某一阈值的特征归类为-1
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

# 机器学习实战之AdaBoost算法
#    https://www.cnblogs.com/zy230530/p/6909288.html
# 构建单层分类器
# 单层分类器是基于最小加权分类错误率的树桩
# 伪代码看起来大致如下：
#   将最小错误率minError设为+∞
#   对数据集中的每一个特征（第一层循环）：
#       对每个步长（第二层循环）：
#           对每个不等号（第三层循环）：
#               建立一棵单层决策树并利用加权数据集对它进行测试
#               如果错误率低于minError，则将当前单层决策树设为最佳单层决策树
#   返回最佳单层决策树
# 遍历stumpClassify()函数所有的可能输入值，并找到数据集上最佳的单层决策树。
# 传入的参数有：数据集，类型标签，权重向量D。
# 返回：决策树，最⼩误差，预测值。
def buildStump(dataArr,classLabels,D):
    # 将数据集和标签列表转为矩阵形式
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    # 用于在特征的所有可能值上进行遍历。
    numSteps = 10.0; 
    # 这个字典用于存储给定权重向量D时所得到的最佳单层决策树的相关信息。
    bestStump = {}; 
    bestClasEst = mat(zeros((m,1)))
    # 在一开始就初始化成正无穷大，之后用于寻找可能的最小错误率。
    minError = inf #init error sum, to +infinity

    # 第一层for循环在数据集的所有特征上遍历。
    for i in range(n):#loop over all dimensions
        # 通过计算最小值和最大值来了解应该需要多大的步长。
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        # 第二层for循环再在这些值上遍历。甚至将阈值设置为整个取值范围之外也是可以的。
        # 遍历各个步长区间
        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
            # 最后一个for循环则是在大于和小于之间切换不等式。
            # 遍历大于和小于两种阈值过滤模式的情况。
            for inequal in ['lt', 'gt']: #go over less than and greater than
                # 从最小值开始，生成每一步的阈值。
                threshVal = (rangeMin + float(j) * stepSize)
                # 在数据集及三个循环变量上调用stumpClassify()函数。
                # 基于这些循环变量，该函数将会返回分类预测结果。
                # 根据生成这一步的阈值和大于小于号。生成数据集中每一个元素的分类预测结果。
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal) # call stump classify with i, j, lessThan
                # 构建一个列向量errArr，先全部设置为一。
                errArr = mat(ones((m,1)))
                # 之后把predictedVals等于labelMat中的真正类别标签值的位置设置为零。
                # 那么predictedVals不等于labelMat中的真正类别标签值就会为一。
                # 把预测正确的元素设置为零。保留分错的数据。
                errArr[predictedVals == labelMat] = 0
                # 计算"加权"的错误率
                # 将错误向量errArr和权重向量D的相应元素相乘并求和，就得到了数值weightedError
                weightedError = D.T*errArr  #calc total error multiplied by D
                # print出所有的值。
                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" \
                #     % (i, threshVal, inequal, weightedError))
                # 将当前的错误率与已有的最小错误率进行对比，
                # 如果当前的值较小，那么就在词典bestStump中保存该单层决策树。
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    # 字典、错误率和类别估计值都会返回给AdaBoost算法。
    return bestStump,minError,bestClasEst

# 完整 AdaBoost 算法的伪代码如下：
#   对每次迭代：
#       利用buildStump()函数找到最佳的单层决策树
#       将最佳单层决策树加入到单层决策树数组
#       计算alpha 
#       计算新的权重向量D
#       更新累计类别估计值
#       如果错误率等于0.0，则退出循环
# 输入参数包括数据集、类别标签以及迭代次数numIt
# 函数名称尾部的DS代表的就是单层决策树（decision stump）
# 它是AdaBoost中最流行的弱分类器。
# 整个代码的原理就是：
#   利用stumpClassify这个弱分类器构建一个带有权重的决策树。
#   达到三个臭皮匠顶个诸葛亮的效果。
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    # 上述算法会输出一个单层决策树的数组，
    # 因此首先需要建立一个新的Python表来对其进行存储。
    weakClassArr = []
    # 得到数据集中的数据点的数目m，
    m = shape(dataArr)[0]
    # 向量D非常重要，它包含了每个数据点的权重。
    # D是一个概率分布向量，一开始的所有元素都会被初始化成1/m。
    D = mat(ones((m,1))/m)   #init D to all equal
    # 记录每个数据点的类别估计累计值。
    aggClassEst = mat(zeros((m,1)))
    # 对每次迭代：该循环运行numIt次或者直到训练错误率为0为止。
    for i in range(numIt):
        # 利用buildStump()函数找到最佳的单层决策树
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#build Stump
        # 打印权重向量
        # print("D:",D.T)
        # 计算单层决策树的系数alpha。该值会告诉总分类器本次单层决策树输出结果的权重。
        # 语句max(error, 1e-16)用于确保在没有错误时不会发生除零溢出。
        # calc alpha, throw in max(error,eps) to account for error=0
        # AdaBoost算法中弱分类器的权重值alpha计算公式如下：
        #    alpha=0.5*ln(1-ε/max(ε,1e-16))
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
        # 将最佳单层决策树加入到单层决策树数组
        # alpha值加入到bestStump字典中，
        bestStump['alpha'] = alpha  
        # 该字典又添加到列表中。该字典包括了分类所需要的所有信息。
        weakClassArr.append(bestStump)      #store Stump Params in Array
        # 打印决策树的预测结果
        # print("classEst: ",classEst.T)
        # 接下来的三行计算新的权重向量D。公式如下：
        # 如果某个样本被正确分类，那么权重更新为：
        #   D(m+1,i)=D(m,i)*exp(-alpha)/sum(D)
        # 如果某个样本被错误分类，那么权重更新为：
        #   D(m+1,i)=D(m,i)*exp(alpha)/sum(D)
        # 预测正确为exp(-alpha),预测错误为exp(alpha)
        # 即增大分类错误样本的权重，减少分类正确的数据点权重
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) #exponent for D calc, getting messy
        # 更新权值向量
        D = multiply(D,exp(expon))                              #Calc New D for next iteration
        D = D/D.sum()
        # 累加当前单层决策树的加权预测
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst
        # print("aggClassEst: ",aggClassEst.T)
        # 求出分类错的样本个数，更新累计类别估计值
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        # 计算错误率
        errorRate = aggErrors.sum()/m
        print("total error: ",errorRate)
        # 如果错误率等于0.0，则退出循环
        if errorRate == 0.0: 
            break
    # 返回弱分类器的组合列表的单层决策树数组。和记录每个数据点的类别估计累计值。
    return weakClassArr,aggClassEst

# 利用训练出的多个弱分类器进行分类的函数。
# 输入是由一个或者多个待分类样例datToClass以及多个弱分类器组成的数组classifierArr。
# 数组classifierArr中的元素由adaBoostTrainDS生成。
def adaClassify(datToClass,classifierArr):
    # 首先将datToClass转换成了一个NumPy矩阵，
    # do stuff similar to last aggClassEst in adaBoostTrainDS
    dataMatrix = mat(datToClass) 
    # 并且得到datToClass中的待分类样例的个数m。
    m = shape(dataMatrix)[0]
    # 后构建一个0列向量aggClassEst
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])#call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst
        # print("adaClassify::aggClassEst : ", aggClassEst)
    return sign(aggClassEst)

# 绘制ROC曲线。
# 函数有两个输入参数，第一个参数是一个NumPy数组或者一个行向量组成的矩阵。
# 该参数代表的则是分类器的预测强度。
# 函数的第二个输入参数是先前使用过的classLabels。
def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = sum(array(classLabels)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    # 按照最小到最大的顺序排列，得到排序索引。
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve is: ",ySum*xStep)
