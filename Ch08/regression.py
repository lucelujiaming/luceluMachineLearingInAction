# coding: utf-8
'''
Created on Jan 8, 2011

@author: Peter
'''
from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

# 计算最佳拟合直线。
# 这个函数的原理如下：
# 平方误差用矩阵表示还可以写做(y-Xw).T * (y-Xw)
# 对w求导，得到X.T * (Y-Xw)，令其等于零。
# 解出w等于 (X.T * X).I * X.T * y
def standRegres(xArr,yArr):
    # 首先读入x和y并将它们保存到矩阵中；
    xMat = mat(xArr); yMat = mat(yArr).T
    # 计算X.T * X
    xTx = xMat.T*xMat
    # 判断它的行列式是否为零，如果行列式为零，那么计算逆矩阵的时候将出现错误。
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    # 变量ws存放的就是回归系数。
    return ws

# 局部加权线性回归：给定x空间中的任意一点，计算出对应的预测值yHat。
def lwlr(testPoint,xArr,yArr,k=1.0):
    # 读入数据并创建所需矩阵，
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    # 创建对角权重矩阵weights 。权重矩阵是一个方阵，阶数等于样本点个数。
    # 也就是说，该矩阵为每个样本点初始化了一个权重。
    weights = mat(eye((m)))
    # 将遍历数据集，
    for j in range(m):    #next 2 lines create weights matrix
        # 计算每个样本点对应的权重值：
        diffMat = testPoint - xMat[j,:]     #
        # 随着样本点与待预测点距离的递增，权重将以指数级衰减
        # 下面是高斯核公式。输入参数k控制衰减的速度。
        # weights = exp(diffMat*diffMat.T/(-2.0*k**2))
        # 这里有一个神奇的类型转换错误。
        # print("diffMat  : ", diffMat)
        # print("diffMat*diffMat.T : ", diffMat*diffMat.T)
        # print("exp(diffMat*diffMat.T/(-2.0*k**2)) : ", exp((diffMat*diffMat.T).sum()/(-2.0*k**2)))
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))[0, 0]
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

# 为数据集中每个点调用lwlr。
def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def lwlrTestPlot(xArr,yArr,k=1.0):  #same thing as lwlrTest except it sorts X first
    yHat = zeros(shape(yArr))       #easier for plotting
    xCopy = mat(xArr)
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i],xArr,yArr,k)
        print("yHat[", i, "] : ", yHat[i])
    return yHat,xCopy

# 计算预测误差的大小
def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()

# 计算回归系数。
# 加入了岭回归以后，回归系数的计算公式将变成：
#     (X.T * X + λ * I ).I * X.T * y
def ridgeRegres(xMat,yMat,lam=0.2):
    # 首先构建矩阵XTX，
    xTx = xMat.T*xMat
    # 然后用lam乘以单位矩阵，作为岭，在矩阵XTX上加一个λI从而使得矩阵非奇异，
    # 进而能对XTX + λI求逆。
    denom = xTx + eye(shape(xMat)[1])*lam
    # 因为如果lambda设定为0的时候一样可能会产生错误，
    # 所以这里仍需要做一个行列式是否为零的检查。
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    # 最后，如果矩阵非奇异就计算回归系数并返回。
    ws = denom.I * (xMat.T*yMat)
    return ws

# 在一组λ上测试结果
def ridgeTest(xArr,yArr):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #to eliminate X0 take mean off of Y
    # 下面展示了数据标准化的过程。
    #regularize X's
    xMeans = mean(xMat,0)   #calc mean then subtract it off
    xVar = var(xMat,0)      #calc variance of Xi then divide by it
    # 所有特征都减去各自的均值并除以方差。
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    # 在30个不同的lambda下调用ridgeRegres()函数。
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    # 将所有的回归系数输出到一个矩阵并返回。
    return wMat

def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

# 前向逐步线性回归算法的实现
# 伪代码如下所示：
#   1. 数据标准化，使其分布满足0均值和单位方差
#   2. 在每轮迭代过程中：
#       3. 设置当前最小误差lowestError为正无穷
#       4. 对每个特征：
#           增大或缩小：
#               改变一个系数得到一个新的W
#               计算新W下的误差
#               如果误差Error小于当前最小误差lowestError：设置Wbest等于当前的W 
#           将W设置为新的Wbest
# 输入包括：
#       输入数据xArr和预测变量yArr。
#       eps，表示每次迭代需要调整的步长；
#       numIt，表示迭代次数。
def stageWise(xArr,yArr,eps=0.01,numIt=100):
    # 1. 数据标准化，使其分布满足0均值和单位方差
    #    先将输入数据转换并存入矩阵中
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    # Can also regularize ys but will get smaller coef
    yMat = yMat - yMean
    # 把特征按照均值为0方差为1进行标准化处理。
    xMat = regularize(xMat)
    m,n=shape(xMat)
    returnMat = zeros((numIt,n)) #testing code remove
    # 创建了一个向量ws来保存w的值
    ws = zeros((n,1)); 
    # 为了实现贪心算法建立了ws的两份副本
    wsTest = ws.copy(); wsMax = ws.copy()
    # 2. 接下来的优化过程需要迭代numIt次，
    for i in range(numIt):
        # 在每次迭代时都打印出w向量，用于分析算法执行的过程和效果。
        print("ws.T : ", ws.T)
        # 3. 误差初始值设为正无穷
        lowestError = inf; 
        # 在所有特征上运行两次for循环
        for j in range(n):
            # 计算增加或减少该特征对误差的影响。
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                # 通过之前的函数rssError()使用平方误差。
                rssE = rssError(yMat.A,yTest.A)
                # 与所有的误差比较后取最小的误差。整个过程循环迭代进行。
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat

#def scrapePage(inFile,outFile,yr,numPce,origPrc):
#    from BeautifulSoup import BeautifulSoup
#    fr = open(inFile); fw=open(outFile,'a') #a is append mode writing
#    soup = BeautifulSoup(fr.read())
#    i=1
#    currentRow = soup.findAll('table', r="%d" % i)
#    while(len(currentRow)!=0):
#        title = currentRow[0].findAll('a')[1].text
#        lwrTitle = title.lower()
#        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
#            newFlag = 1.0
#        else:
#            newFlag = 0.0
#        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
#        if len(soldUnicde)==0:
#            print("item #%d did not sell" % i)
#        else:
#            soldPrice = currentRow[0].findAll('td')[4]
#            priceStr = soldPrice.text
#            priceStr = priceStr.replace('$','') #strips out $
#            priceStr = priceStr.replace(',','') #strips out ,
#            if len(soldPrice)>1:
#                priceStr = priceStr.replace('Free shipping', '') #strips out Free Shipping
#            print("%s\t%d\t%s" % (priceStr,newFlag,title))
#            fw.write("%d\t%d\t%d\t%f\t%s\n" % (yr,numPce,newFlag,origPrc,priceStr))
#        i += 1
#        currentRow = soup.findAll('table', r="%d" % i)
#    fw.close()
    
from time import sleep
import json
# import urllib2
import urllib.request
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urllib.request.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else: newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if  sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except: print('problem with item %d' % i)
    
def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)

# 前两个参数lgX和lgY存有数据集中的X和Y值的list对象，
# 默认lgX和lgY具有相同的长度。
# 第三个参数numVal是算法中交叉验证的次数，如果该值没有指定，就取默认值10。
def crossValidation(xArr,yArr,numVal=10):
    # 首先计算数据点的个数m
    m = len(yArr)                           
    indexList = range(m)
    errorMat = zeros((numVal,30))#create error mat 30columns numVal rows
    for i in range(numVal):
        # 创建好了训练集和测试集容器
        trainX=[]; trainY=[]
        testX = []; testY = []
        # 创建了一个list并进行混洗（shuffle），
        # 因此可以实现训练集或测试集数据点的随机选取。
        random.shuffle(indexList)
        # 将数据集的90%分割成训练集，其余10%为测试集，并将二者分别放入对应容器中。
        for j in range(m):#create training set based on first 90% of values in indexList
            if j < m*0.9: 
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        # 建立一个新的矩阵wMat来保存岭回归中的所有回归系数。
        wMat = ridgeTest(trainX,trainY)    #get 30 weight vectors from ridge
        # 在上述测试集上用30组回归系数来循环测试回归效果。
        for k in range(30):#loop over all of the ridge estimates
            matTestX = mat(testX); matTrainX=mat(trainX)
            meanTrain = mean(matTrainX,0)
            varTrain = var(matTrainX,0)
            matTestX = (matTestX-meanTrain)/varTrain #regularize test with training params
            yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)#test ridge results and store
            # 用函数rssError()计算误差，并将结果保存在errorMat中。
            errorMat[i,k]=rssError(yEst.T.A,array(testY))
            #print(errorMat[i,k])
    # 还原使用了数据标准化还原数据
    meanErrors = mean(errorMat,0)#calc avg performance of the different ridge weight vectors
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors==minMean)]
    #can unregularize to get model
    #when we regularized we wrote Xreg = (x-meanX)/var(x)
    #we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
    xMat = mat(xArr); yMat=mat(yArr).T
    meanX = mean(xMat,0); varX = var(xMat,0)
    # 计算这些误差估计值的均值
    unReg = bestWeights/varX
    print("the best model from Ridge Regression is:\n",unReg)
    print("with constant term: ",-1*sum(multiply(meanX,unReg)) + mean(yMat))