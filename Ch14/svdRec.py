'''
Created on Mar 8, 2011

@author: Peter
'''
from numpy import *
from numpy import linalg as la

def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]
    
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]
# 各种相似度的计算方法
# 欧氏距离
def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))
# 皮尔逊相关系数
def pearsSim(inA,inB):
    if len(inA) < 3 : return 1.0
    return 0.5+0.5*corrcoef(inA, inB, rowvar = 0)[0][1]
# 余弦相似度
def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

# 计算在给定相似度计算方法的条件下，用户对物品的估计评分值。
# 参数包括数据矩阵、用户编号、物品编号和相似度计算方法。
def standEst(dataMat, user, simMeas, item):
    # 假设这里的数据矩阵为中，即行对应用户、列对应物品。
    # 首先会得到数据集中的物品数目，
    n = shape(dataMat)[1]
    # 然后对两个后面用于计算估计评分值的变量进行初始化。
    simTotal = 0.0; ratSimTotal = 0.0
    # 循环大体上是对用户评过分的每个物品进行遍历，
    # 并将它和其他物品进行比较。
    for j in range(n):
        userRating = dataMat[user,j]
        # 如果某个物品评分值为0，就意味着用户没有对该物品评分，
        if userRating == 0: 
            # 跳过这个物品。
            continue
        # 变量overlap给出的是两个物品当中已经被评分的那个元素。
        overLap = nonzero(logical_and(dataMat[:,item].A>0, \
                                      dataMat[:,j].A>0))[0]
        # 如果两者没有任何重合元素，则相似度为0且中止本次循环。
        if len(overLap) == 0: 
            similarity = 0
        # 但是如果存在重合的物品，则基于这些重合物品计算相似度。
        else: 
            similarity = simMeas(dataMat[overLap,item], \
                      dataMat[overLap,j])
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        # 相似度会不断累加，
        simTotal += similarity
        # 每次计算时还考虑相似度和当前用户评分的乘积。
        ratSimTotal += similarity * userRating
    if simTotal == 0: 
        return 0
    # 最后，通过除以所有的评分总和，对上述相似度评分的乘积进行归一化。
    else: return ratSimTotal/simTotal

# 基于SVD的评分估计函数。
# 在recommend()中，这个函数用于替换对standEst()的调用，
# 该函数对给定用户给定物品构建了一个评分估计值。
def svdEst(dataMat, user, simMeas, item):
    # 首先会得到数据集中的物品数目，
    n = shape(dataMat)[1]
    # 然后对两个后面用于计算估计评分值的变量进行初始化。
    simTotal = 0.0; ratSimTotal = 0.0
    # 对数据集进行了SVD分解。
    U,Sigma,VT = la.svd(dataMat)
    # 用这些奇异值构建出一个对角矩阵
    # arrange Sig4 into a diagonal matrix
    Sig4 = mat(eye(4)*Sigma[:4]) 
    # 利用U矩阵将物品转换到低维空间中
    # create transformed items
    xformedItems = dataMat.T * U[:,:4] * Sig4.I  
    # 循环大体上是对用户评过分的每个物品行的所有元素上进行遍历，
    # 并将它和其他物品进行比较。
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: continue
        # 这里的相似度计算是在低维空间下进行的。
        # 相似度的计算方法也会作为一个参数传递给该函数。
        similarity = simMeas(xformedItems[item,:].T,\
                             xformedItems[j,:].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        # 对相似度求和，
        simTotal += similarity
        # 同时对相似度及对应评分值的乘积求和
        ratSimTotal += similarity * userRating
    if simTotal == 0: 
        return 0
    else: 
        # 这些值返回之后则用于估计评分的计算。
        return ratSimTotal/simTotal

# 基于物品相似度的推荐引擎
# 产生了最高的N个推荐结果。默认值为3。
# 一共五个参数：
#   数据集，给定的用户。N的大小。
#   相似度计算方法和估计方法。
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    # 第一件事就是对给定的用户建立一个未评分的物品列表
    unratedItems = nonzero(dataMat[user,:].A==0)[1]#find unrated items 
    # 如果不存在未评分物品，那么就退出函数；
    if len(unratedItems) == 0: 
        return 'you rated everything'
    itemScores = []
    # 否则，在所有的未评分物品上进行循环。
    for item in unratedItems:
        # 对每个未评分物品，则通过调用standEst()来产生该物品的预测得分。
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        # 该物品的编号和估计得分值会放在一个元素列表itemScores中。
        itemScores.append((item, estimatedScore))
    # 最后按照估计得分，对该列表进行排序并返回。
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

# 打印矩阵。由于矩阵包含了浮点数，因此必须定义浅色和深色。
# 这里通过一个阈值来界定，后面也可以调节该值。
def printMat(inMat, thresh=0.8):
    # 该函数遍历所有的矩阵元素，
    for i in range(32):
        for k in range(32):
            # 当元素大于阈值时打印1，否则打印0。
            if float(inMat[i,k]) > thresh:
                print(1, end="")
            else: print(0, end="")
        print('', end="")

# 使用SVD来对数据降维的图像压缩函数：
def imgCompress(fileName, numSV=3, thresh=0.8):
    # 构建了一个列表，
    myl = []
    # 然后打开文本文件，并从文件中以数值方式读入字符。
    # for line in open('0_5.txt').readlines():
    for line in open(fileName).readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    # 在矩阵调入之后，
    myMat = mat(myl)
    print("****original matrix******")
    # 我们就可以在屏幕上输出该矩阵了。
    printMat(myMat, thresh)
    # 接下来就开始对原始图像进行SVD分解并重构图像。
    U,Sigma,VT = la.svd(myMat)
    # 通过将Sigma重新构成SigRecon来实现这一点。
    # Sigma是一个对角矩阵，因此需要建立一个全0矩阵，
    SigRecon = mat(zeros((numSV, numSV)))
    # 然后将前面的那些奇异值填充到对角线上。
    for k in range(numSV):#construct diagonal matrix from vector
        SigRecon[k,k] = Sigma[k]
    # 最后，通过截断的U和VT矩阵，用SigRecon得到重构后的矩阵，
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:]
    print("****reconstructed matrix using %d singular values******" % numSV)
    # 该矩阵通过printMat()函数输出。
    printMat(reconMat, thresh)