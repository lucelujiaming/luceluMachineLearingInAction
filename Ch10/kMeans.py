# coding: utf-8
'''
Created on Feb 16, 2011
k Means Clustering for Ch10 of Machine Learning in Action
@author: Peter Harrington
'''
from numpy import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 数据集每行包括两个元素。使用\t分割。
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

# 计算两个向量的欧式距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)
# 为给定数据集构建一个包含k个随机质心的集合。
# 也就是一个k行，和数据集列相等的一个矩阵。里面会填充上每一列min到max之间的值。
def randCent(dataSet, k):
    # 得到列大小。对于我们现在使用的数据集来说，等于2。
    n = shape(dataSet)[1]
    # 创建质心矩阵。初始化为全零。
    centroids = mat(zeros((k,n)))#create centroid mat
    # create random cluster centers, within bounds of each dimension
    # 下面一列一列的处理。
    for j in range(n):
        # 先找到每一列的最小值。对于二维数据来说就是一个数字。
        minJ = min(dataSet[:,j]) 
        # 再找到每一列的最大值。之后计算最大值和最小值的差值。
        rangeJ = float(max(dataSet[:,j]) - minJ)
        # 使用最小值和差值生成一列数据。
        # 方法就是最小值 + 差值 * random(1)
        # 这样生成的数字必然位于最小值和最大值之间。
        # 这列数据的行数由输入的k指定。
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

# 算法思想：
#   创建k个质心，然后将每个点分配到最近的质心，再重新计算质心。
#   这个过程重复数次，直到数据点的簇分配结果不再改变为止。
# 算法伪代码表示如下：
#   创建k个点作为起始质心（经常是随机选择）
#   当任意一个点的簇分配结果发生改变时
#       对数据集中的每个数据点
#           对每个质心
#               计算质心与数据点之间的距离
#           将数据点分配到距其最近的簇
#       对每一个簇，计算簇中所有点的均值并将均值作为质心
# 4个输入参数。
#       数据集
#       簇的数目
#       用来计算距离和创建初始质心的函数(可选)
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    # 一开始确定数据集中数据点的总数，
    m = shape(dataSet)[0]
    # 然后创建一个矩阵来存储每个点的簇分配结果。
    # 簇分配结果矩阵clusterAssment包含两列：一列记录簇索引值，第二列存储误差。
    clusterAssment = mat(zeros((m,2)))#create mat to assign data points 
                                      #to a centroid, also holds SE of each point
    # 先计算出来一个包含k个随机质心的集合。
    # 数据一共有k行。
    centroids = createCent(dataSet, k)
    clusterChanged = True
    # 按照上述方式（即计算质心 - 分配 - 重新计算）反复迭代，
    while clusterChanged:
        clusterChanged = False
        # 选取数据集的一行。
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf; minIndex = -1
            # 接下来遍历所有数据找到距离每个点最近的质心。
            # 首先选取随机质心的一行。
            for j in range(k):
                # 使用distMeas函数指针计算距离，默认距离函数是distEclud()。
                # 针对数据集的某一行，计算和前面得到的每一条质心数据的距离。
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                # 如果发现对于数据集的这一行来说，
                # 计算出来的距离比之前计算出来的其他条质心数据的距离。
                # 那么就记录下来。
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            # 执行到这里。上面的循环结束以后，我们得到了和数据集的某一行距离最近的一条质心数据。
            # 如果所有数据点的簇分配结果发生改变，
            # 说明上面的循环找到了比当前已经选择的质心数据更好的一条。
            # 则更新质心变更标志。
            if clusterAssment[i,0] != minIndex: 
                clusterChanged = True
            # 并且更新质心信息。注意簇分配结果矩阵clusterAssment包含两列：
            # 一列记录簇索引值，第二列存储误差，也就是距离的平方。
            clusterAssment[i,:] = minIndex,minDist**2
        # 执行到这里。上面的循环结束以后，我们得到了数据集的每一行距离最近的一条质心数据。
        # 保存在clusterAssment里面。
        print(centroids)
        # 遍历所有质心并更新它们的取值。
        for cent in range(k):#recalculate centroids
            # 通过数组过滤来获得给定簇的所有点；
            # get all the point in this cluster
            # 首先clusterAssment里面的第一列记录了簇索引值，因此上：
            #     clusterAssment[:,0].A==cent
            # 给出了数据集的那些行使用了当前这一行质心。
            # 因为这个簇索引值其实是和数据集的行号一一对应的。
            # 通过数组过滤的方法，我们得到了数据集中使用第k行质心的所有行。
            # 进而通过[0]得到数据集中这些行的第一列。
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            # 然后计算所有点的均值，选项axis = 0表示沿矩阵的列方向进行均值计算
            # 对上面使用第k行质心的所有行的第一列求均值。作为新的质心。
            # 因为这些行既然使用了同一个质心，那么他们有必要通过求均值的方法选出这些行的新的质心。
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean 
        # 如果上面的clusterChanged未被置为真。那么这就是最后一次更新质心。函数将退出。
        # 如果我们更新了质心。那么我们就需要开始新一轮的循环。看看能不能继续找到更好的质心。
    # 返回所有的类质心与点分配结果。
    return centroids, clusterAssment

# 二分 K-均值算法：
#   将所有点看成一个簇，然后将该簇一分为二。
#   之后选择其中一个簇调用kMeans继续进行划分，
#   选择哪一个簇进行划分取决于对其划分是否可以最大程度降低SSE的值。
#   上述基于SSE的划分过程不断重复，直到得到用户指定的簇数目为止。
# 伪代码形式如下：
#   当簇数目小于k时
#       对于每一个簇
#           计算总误差
#           在给定的簇上面进行K-均值聚类（k=2）
#           计算将该簇一分为二之后的总误差
#   选择使得误差最小的那个簇进行划分操作
def biKmeans(dataSet, k, distMeas=distEclud):
    # 首先创建一个矩阵来存储数据集中每个点的簇分配结果及平方误差
    # 获取数据集行数。该数据集为m行n列。
    m = shape(dataSet)[0]
    # 创建一个矩阵，行数和数据集相同，但是有两列。
    clusterAssment = mat(zeros((m,2)))
    # 求数据集所有行的平均值。返回一个一行n列的列表。作为初始的质心。
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    # 把这个质心加入质心列表。这个列表目前只有一个元素。
    # 后续发现的质心会加入这个列表中。直到这个列表的个数满足输入参数要求的k个。
    centList =[centroid0] #create a list with one centroid
    # 计算整个数据集的质心，并使用一个列表来保留所有的质心
    # 得到上述初始的质心之后，可以遍历数据集中所有点来计算每个点到质心的误差值。
    for j in range(m):#calc initial Error
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    # 该循环会不停对簇进行划分，直到得到想要的簇数目为止。
    # 可以通过考察簇列表中的值来获得当前簇的数目。然后遍历所有的簇来决定最佳的簇进行划分。
    # 只有质心列表中的质心数量达到数量要求才会退出。
    while (len(centList) < k):
        # 一开始将最小SSE置设为无穷大
        lowestSSE = inf
        # 遍历簇列表centList中的每一个簇。
        for i in range(len(centList)):
            # get the data points currently in cluster i
            # 通过数组过滤的方法，我们得到了数据集中使用第k行质心的所有行。
            # 进而通过[0]得到数据集中这些行的第一列。这个逻辑在kMeans中见过了。
            # 也就是将该簇中的所有点看成一个小的数据集 ptsInCurrCluster。
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            # 把数据集中用到这个质心的数据输入到函数kMeans()中进行处理。
            # K-均值算法会生成2个质心（簇），同时给出每个簇的误差值。
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            # 这些误差与剩余数据集的误差之和作为本次划分的误差。
            # compare the SSE to the currrent minimum
            # 簇分配结果矩阵splitClustAss包含两列：一列记录簇索引值，第二列存储误差。
            # 这里得到kMeans返回的两个簇的误差值的和。
            sseSplit = sum(splitClustAss[:,1])
            # 这里得到数据集 ptsInCurrCluster不属于这两个簇的数据的误差值之和。
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            # 把kMeans返回的两个簇的误差值与剩余数据集的误差值之和作为本次划分的误差。
            # 如果该划分的SSE值最小，则本次划分被保存。
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 程序运行到这里，我们已经遍历了簇列表centList中的每一个簇，找到了一个要划分的簇。
        # 接下来就要实际执行划分操作。划分操作很容易，
        # 只需要将要划分的簇中所有点的簇分配结果进行修改即可。
        # 将这些簇编号修改为划分簇及新加簇的编号，该过程可以通过两个数组过滤器来完成。
        # change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) 
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print('the bestCentToSplit is: ',bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        # 最后，新的质心会被添加到centList中。
        # replace a centroid with two best centroids 
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]
        centList.append(bestNewCents[1,:].tolist()[0])
        # 新的簇分配结果被更新，
        # reassign new clusters, and SSE
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss
    # 当while循环结束时，函数返回质心列表与簇分配结果。
    return mat(centList), clusterAssment

import urllib
import json
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  #create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'#JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params      #print(url_params
    print(yahooApi)
    c=urllib.urlopen(yahooApi)
    return json.loads(c.read())

from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else: print("error fetching")
        sleep(1)
    fw.close()
    
def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 #pi is imported with numpy

def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()
