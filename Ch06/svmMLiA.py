# coding: utf-8
'''
Created on Nov 4, 2010
Chapter 5 source file for Machine Learing in Action
@author: Peter
'''
from numpy import *
from time import sleep

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

# 两个参数值，其中i是第一个alpha的下标，m是所有alpha的数目。
# 只要函数值不等于输入值i，函数就会进行随机选择。
# 也就是说，随机选择一个j。如果再次随机到i，就继续随机，直到不等于i。
def selectJrand(i,m):
    # 先让j等于i
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

# 用于调整大于H或小于L的alpha值。
def clipAlpha(aj,H,L):
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj

# 伪代码大致如下：
#     创建一个alpha向量并将其初始化为0向量
#     当迭代次数小于最大迭代次数时（外循环）
#         对数据集中的每个数据向量（内循环）：
#             如果该数据向量可以被优化：
#                 随机选择另外一个数据向量
#                 同时优化这两个向量
#                 如果两个向量都不能被优化，退出内循环
#         如果所有向量都没被优化，增加迭代数目，继续下一次循环
# 有5个输入参数，分别是：
#   数据集、类别标签、常数C、容错率和取消前最大的循环次数。
# 这个算法的原理是这样的。
# 首先我们假设平面上有两堆点。一堆属于A，一堆属于B。我们想做一条直线把这两堆点分开。
# 最优的直线必然满足下面的条件就是：
#    平面上的这两堆点对这根线做垂线，这些垂线之和最小。
# 因此上我们首先需要计算这两堆点的垂线长度。
# 根据高中数学，假设当P（x0,y0），直线L的解析式为y=kx+b时，
# 则点P到直线L的距离为：
#    ｜kx0 - By0 + b｜/ sqrt(k^2 + 1)
# 同时为了方便处理，可以如下设定。即：
#    位于数据点正方向的类型标签为1，而位于数据点负方向的类型标签为-1。
# 因此上，上面的问题就变成：
#    label * (kx0 - y0 + b) / sqrt(k^2 + 1)
# 分母其实可以计算进去，进一步有：
#    label * (Kx0 - Ay0 + B)
# 之后把这个推广到高维。上面公式中的label，k，b都变成矩阵。
#    labelMat * (wTx + bMat)
# 因此上，上面的问题就变成求wT和b，使得sum(labelMat * (wTx + bMat))最小。
# 也就是: arg max{ min(label * (wT + b) / |w|) }
# 上面的这个问题是一个非常大的二次规划（QP）优化问题。
# 为了解决这个问题，提出了SMO算法。可以把这个大的QP问题分解为一系列最小的QP问题。
#      SMO在每一步仅选择两个拉格朗日乘数进行联合优化。
#      其优势在于，两个拉格朗日乘数的优化可以通过解析方法完成，计算更快、更精确。
#      此外，由于 SMO 完全不需要额外的矩阵存储，即使大规模的训练问题也可以在个人电脑的内存中运行。
# 把上面的公式转换为数据点格式以后，使用拉格朗日乘子法可以得到书上92页的优化目标函数。
# 但是，其实这个函数主要的逻辑就是随机找两个向量进行联合优化。
# 因为SMO算法则是把对整个α的优化转化为对每一对αi的优化。
# 用到的SMO算法由两个部分组成：
# 一种求解两个拉格朗日乘数的解析方法，以及一种选择要优化的乘数的启发式方法。
# 前者给出了关于L，H，eta，b1，b2的计算公式。
# 后者的一部分就是这个两层循环的结构。另外的部分在改进函数中。
# 详细内容可以参见SMO的论文。对照代码和论文中的公式就可以了。
# 论文下载方法，在这个链接中搜索"Sequential minimal optimization"即可：
#   https://www.microsoft.com/en-us/research/search/
# 对应中文版讲解如下：
#   https://blog.csdn.net/adamding1999/article/details/107372575
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    # 将数据集转换为NumPy矩阵。
    dataMatrix = mat(dataMatIn)
    # 将类别标签转换为NumPy矩阵。
    labelMat = mat(classLabels).transpose()
    # 获得矩阵大小。
    b = 0; m,n = shape(dataMatrix)
    # 构建一个alpha列矩阵，矩阵中元素都初始化为0，
    alphas = mat(zeros((m,1)))
    # 并建立一个iter变量。该变量存储的则是在没有任何alpha改变的情况下遍历数据集的次数。
    iter = 0
    # 当该变量达到输入值maxIter时，函数结束运行并退出。
    while (iter < maxIter):
        # 每次循环当中，将alphaPairsChanged先设为0，然后再对整个集合顺序遍历。
        # 变量alphaPairsChanged用于记录alpha是否已经进行优化。
        alphaPairsChanged = 0
        # 如果alpha可以更改进入优化过程。
        # 为了是固定其中一个因子而最大化其他因子，
        # 下面一行一行对dataMatrix进行计算。
        for i in range(m):
            # fXi就是我们通过计算，预测出来的类别。
            # 计算公式就是我们上面提到的label * (wT + b) / |w|。
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            # 基于这个实例的预测结果和真实结果的比对，就可以计算误差Ei。
            # 如果误差很大，那么可以对该数据实例所对应的alpha值进行优化。
            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions
            # 在if语句中，不管是正间隔还是负间隔都会被测试。
            # 并且在该if语句中，也要同时检查alpha值，以保证其不能等于0或C。
            # 保证不在边界上。
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) \
            or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                # 如果该数据向量可以被优化，利用辅助函数来随机选择第二个alpha值
                j = selectJrand(i,m)
                # 计算这个alpha值的误差
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                # 为alphaIold和alphaJold分配新的内存
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                # 开始计算L和H，它们用于保证alpha在0与C之间。
                # 这个的计算公式参见SMO的论文。
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: 
                    print("L==H"); continue
                # Eta是alpha[j]的最优修改量。使用下面这个很长的代码计算得到。
                # 这个Eta的计算公式也参见SMO的论文。
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T \
                          - dataMatrix[i,:]*dataMatrix[i,:].T \
                          - dataMatrix[j,:]*dataMatrix[j,:].T
                # 如果eta为0，那就是说需要退出for循环的当前迭代过程。
                if eta >= 0: 
                    print("eta>=0"); continue
                # 于是，可以计算出一个新的alpha[j]，然后利用程序清单6-1中的辅助函数以及L与H值对其进行调整。
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                # 要检查alpha[j]是否有轻微改变。如果是的话，就退出for循环。
                if (abs(alphas[j] - alphaJold) < 0.00001): 
                    print("j not moving enough"); continue
                # 对i进行修改，修改量与j相同，但方向相反
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                                                                        #the update is in the oppostie direction
                # 在对alpha[i]和alpha[j]进行优化之后，给这两个alpha值设置一个常数项b 。
                # 这个b1的计算公式也参见SMO的论文。
                b1 = b - Ei- \
                    labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - \
                    labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- \
                    labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - \
                    labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): 
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): 
                    b = b2
                else: 
                    b = (b1 + b2)/2.0
                # 如果程序执行到for循环的最后一行都不执行continue语句，
                # 那么就已经成功地改变了一对alpha，可以增加alphaPairsChanged的值。
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
        # 在for循环之外，需要检查alpha值是否做了更新，如果有更新则将iter设为0后继续运行程序。
        if (alphaPairsChanged == 0): 
            iter += 1
        else: 
            iter = 0
        # 只有在所有数据集上遍历maxIter次，且不再发生任何alpha修改之后，程序才会停止并退出while循环。
        print("iteration number: %d" % iter)
    return b,alphas

# 用于进行映射转换的核函数
def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m,n = shape(X)
    # 首先构建出了一个列向量，然后检查元组以确定核函数的类型。
    K = mat(zeros((m,1)))
    # 根据所用核函数类型进行分别处理。
    if kTup[0]=='lin': 
        K = X * A.T   #linear kernel
    elif kTup[0]=='rbf':
        # 在for循环中对于矩阵的每个元素计算高斯函数的值。
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        # 在for循环结束之后，我们将计算过程应用到整个向量上去。
        K = exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
    else: 
        # 如果遇到一个无法识别的元组，程序就会抛出异常
        raise NameError('Houston We Have a Problem -- \
                        That Kernel is not recognized')
    return K

class optStruct:
    # kTup: 一个包含核函数信息的元组
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        # 增加了一个m×2的矩阵成员变量eCache之外。
        # eCache的第一列给出的是eCache是否有效的标志位，而第二列给出的是实际的E值。
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag
        self.K = mat(zeros((self.m,self.m)))
        # 核函数矩阵K先被构建，然后再通过调用函数kernelTrans()进行填充。全局的K值只需计算一次。
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

# 计算E值并返回。也就是W.T * x + b
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

# 用于选择第二个alpha或者说内循环的alpha值，以保证在每次优化中采用最大步长。 
# this is the second choice -heurstic, and calcs Ej     
def selectJ(i, oS, Ei):
    maxK = -1; maxDeltaE = 0; Ej = 0
    # 首先将输入值Ei在缓存中设置成为有效的。
    # 这里的有效（valid）意味着它已经计算好了。
    # set valid #choose the alpha that gives the maximum delta E
    oS.eCache[i] = [1,Ei]  
    # 构建出了一个非零表。这个列表中包含以输入列表为目录的列表值。
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    # 在所有的值上进行循环并选择其中使得改变最大的那个值。
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i:
                continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            # 根据SMO的论文，第二个变量的标准是希望能使alpha2有足够大的变化。
            # 而alpha2是依赖于|E1−E2|，
            # 为了加快计算的速度，做简单的就是选择|E1−E2|最大时的alpha2。
            # 选择其中使得改变最大的那个值。
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        # 返回改变最大的那个值。
        return maxK, Ej
    # 如果这是第一次循环的话，那么就随机选择一个alpha值。
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

# 计算误差值并存入缓存当中。在对alpha值进行优化之后会用到这个值。
def updateEk(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

# 寻找决策边界的优化例程
def innerL(i, oS):
    # 1. 根据输入的i计算E值。
    Ei = calcEk(oS, i)
    # 2. 如果E值在范围以内。
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
       ((oS.labelMat[i]*Ei >  oS.tol) and (oS.alphas[i] >    0)):
        # 使用SelectJ()而不是selectJrand()来选择第二个alpha的值
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        # 3. 复制选出来的i元素和j元素。
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        # 4. 将alpha[j]调整到0到C之间。
        #    常数C给出的是不同优化问题的权重。常数C一方面要保障所有样例的间隔不小于1.0，
        #    另一方面又要使得分类间隔要尽可能大，并且要在这两方面之间平衡。
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        # 5. 如果L和H相等，就不做任何改变，退出并进行下一轮的smoP循环。
        if L==H: 
            print("L==H"); 
            return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
        if eta >= 0: 
            print("eta>=0"); 
            return 0
        # 6. alpha[i]和alpha[j]同样进行改变，虽然改变的大小一样，但是改变的方向正好相反。
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): 
            print("j not moving enough"); 
            return 0
        # update i by the same amount as j
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        # added this for the Ecache                    
        # the update is in the oppostie direction
        updateEk(oS, i) 
        # 7. 在对alpha[i]和alpha[j]进行优化之后，给这两个alpha值设置一个常数项b。
        # 这个地方有两个计算版本。一个是不采用核函数K的版本。一个是使用了核函数K的版本。
        # 在这里，你可以把它想象成为另外一种距离计算的方法。由kernelTrans函数生成。
        # b1 = oS.b - Ei \
        #       - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:] * oS.X[i,:].T \
        #       - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:] * oS.X[j,:].T
        b1 = oS.b - Ei \
                - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] \
                - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        # b2 = oS.b - Ej \
        #       - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:] * oS.X[j,:].T \
        #       - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:] * oS.X[j,:].T
        b2 = oS.b - Ej \
                - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j] \
                - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): 
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): 
            oS.b = b2
        else: 
            oS.b = (b1 + b2)/2.0
        return 1
    else: 
        return 0

# 完整版Platt SMO的外循环代码
# 和smoSimple相比，加入了很多启发性的步骤。
# 这就是SMO论文中的另一个部分：一种选择要优化的乘数的启发式方法。
# 包括下面的几个改进点：
#   首先把E值计算抽出来，独立成一个calcEk函数。
#。 误差值计算也抽出来，独立成一个updateEk函数。
#   之后是实现了selectJ函数，修改了第二个alpha的选择方法。
#   而第一个alpha的选择方法也有变化。
#   b1, b2的计算也引入了核函数。
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    # 开始构建一个数据结构来容纳所有的数据，
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    # 然后需要对控制函数退出的一些变量进行初始化。
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    # 当迭代次数超过指定的最大值，或者遍历整个集合都未对任意alpha对进行修改时，就退出循环。
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        # 1. 遍历所有的值
        # ，一开始的for循环在数据集上遍历任意可能的alpha 。
        if entireSet:   #go over all
            for i in range(oS.m):   
                # 我们通过调用innerL()来选择第二个alpha，并在可能时对其进行优化处理。
                # 如果有任意一对alpha值发生改变，那么会返回 1。     
                alphaPairsChanged += innerL(i,oS)
            print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        # 2. 遍历非边界值
        # 第二个for循环遍历所有的非边界alpha值，也就是不在边界0或C上的值
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
            print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        # 接下来，我们对for循环在非边界循环和完整遍历之间进行切换，并打印出迭代次数。最后程序将会返回常数b和alpha值。
        if entireSet:
            entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): 
            entireSet = True  
        print("iteration number: %d" % iter)
    return oS.b,oS.alphas

# 使用计算出来的alpha值进行分类
def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    # 大部分alpha值为0。而非零alpha所对应的也就是支持向量。
    # 虽然上述for循环遍历了数据集中的所有数据，但是最终起作用只有支持向量。
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

# 该输入参数是高斯径向基函数中的一个用户定义变量。
def testRbf(k1=1.3):
    # 首先，程序从文件中读入数据集，
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    # 然后在该数据集上运行Platt SMO算法，其中核函数的类型为'rbf'。
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    # 建立了数据的矩阵副本，
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    # 并且找出那些非零的alpha值，
    svInd=nonzero(alphas.A>0)[0]
    # 从而得到所需要的支持向量；
    sVs=datMat[svInd] #get matrix of only support vectors
    labelSV = labelMat[svInd];
    print("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    # 整个代码中最重要的是for循环开始的那两行，它们给出了如何利用核函数进行分类。
    for i in range(m):
        # 利用结构初始化方法中使用过的kernelTrans()函数，得到转换后的数据。
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        # 然后，再用其与前面的alpha及类别标签值求积。
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        # 如果计算结果和标签不同方向，说明分类错误。
        if sign(predict)!=sign(labelArr[i]): 
            errorCount += 1
    # 打印错误率。
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    # 与第一个for循环相比，第二个for循环仅仅只有数据集不同，后者采用的是测试数据集。
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print("the test error rate is: %f" % (float(errorCount)/m))  
    
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels    

# 这里的函数元组kTup是输入参数，而在testRbf()中默认的就是使用rbf核函数。
def testDigits(kTup=('rbf', 10)):
    # 首先，程序从文件中读入数据集，
    dataArr,labelArr = loadImages('trainingDigits')
    # 然后在该数据集上运行Platt SMO算法，其中核函数的类型为'rbf'
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    # 建立了数据的矩阵副本，
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    # 并且找出那些非零的alpha值，
    svInd=nonzero(alphas.A>0)[0]
    # 从而得到所需要的支持向量；
    sVs=datMat[svInd] 
    labelSV = labelMat[svInd];
    print("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    # 整个代码中最重要的是for循环开始的那两行，它们给出了如何利用核函数进行分类。
    for i in range(m):
         # 利用结构初始化方法中使用过的kernelTrans()函数，得到转换后的数据。
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        # 然后，再用其与前面的alpha及类别标签值求积。
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        # 如果计算结果和标签不同方向，说明分类错误。
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadImages('testDigits')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    # 与第一个for循环相比，第二个for循环仅仅只有数据集不同，后者采用的是测试数据集。
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print("the test error rate is: %f" % (float(errorCount)/m))


'''#######********************************
Non-Kernel VErsions below
'''#######********************************

class optStructK:
    def __init__(self,dataMatIn, classLabels, C, toler):  # Initialize the structure with the parameters 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag
        
def calcEkK(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek
        
def selectJK(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEkK(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]
        
def innerLK(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print("L==H"); return 0
        eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        if eta >= 0: print("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

def smoPK(dataMatIn, classLabels, C, toler, maxIter):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True  
        print("iteration number: %d" % iter)
    return oS.b,oS.alphas