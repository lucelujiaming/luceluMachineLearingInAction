# coding: utf-8
'''
Created on Oct 19, 2010

@author: Peter
'''
from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec
# 创建一个包含在所有文档中出现的不重复词的列表，
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)
# 输入参数为词汇表及某个文档，输出的是文档向量。
def setOfWords2Vec(vocabList, inputSet):
    # 创建一个和词汇表等长的向量，并将其元素都设置为0。
    returnVec = [0]*len(vocabList)
    # 遍历文档中的所有单词，
    for word in inputSet:
        # 如果出现了词汇表中的单词，
        if word in vocabList:
            # 则将输出的文档向量中的对应值设为1。
            returnVec[vocabList.index(word)] = 1
        else: 
            # 一切都顺利的话，就不需要检查某个词是否还在vocabList中
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

# 从词向量计算概率的函数的伪代码如下：
# 下面提到的所谓的词条，其实就是类型标签。
# 计算每个类别中的文档数目
# 对每篇训练文档：
#   对每个类别：
#       如果词条出现文档中 → 增加该词条的计数值
#       增加所有词条的计数值
#   对每个类别：
#       对每个词条：
#           将该词条的数目除以总词条数目得到条件概率
#   返回每个类别的条件概率
# 输入参数为文档矩阵trainMatrix，
#      以及由每篇文档类别标签所构成的向量trainCategory。
# 
# trainMatrix是一个二维数组。包含多个向量。向量个数为词条个数。
# 向量内容是这个词条在整体数据集中出现的概率。
# 这就构成了一个文档矩阵，每一个元素的列坐标表示数据集单词序号。
# 行坐标表示这个单词对应的词条序号。对于位于(n, m)位置的元素。
# 如果为零，表明数据集的第n个单词在m号词条中未出现。
# 否则如果为1，就表明数据集的第n个单词在m号词条中出现。
def trainNB0(trainMatrix,trainCategory):
    # 得到文档矩阵的向量个数。这个其实就是词条，或者说是类型标签的个数。
    numTrainDocs = len(trainMatrix)
    # 得到整体数据集的单词个数。
    numWords = len(trainMatrix[0])
    # 计算类型标签的abusive概率，因为1代表abusive。
    # 因此上对于[0,1,0,1,0,1]这个list，abusive概率为0.5。
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # 要计算多个概率的乘积以获得文档属于某个类别的概率。
    # 如果其中一个概率值为0，那么最后的乘积也为0。
    # 为降低这种影响，可以将所有词的出现数初始化为1，并将分母初始化为2。
    p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones() 
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    # 遍历训练集trainMatrix中的每一个类型标签对应的文档向量。
    for i in range(numTrainDocs):
        # 一旦某个词语（侮辱性或正常词语）在某一文档中出现，
        # 如果这个类型标签是侮辱性。
        if trainCategory[i] == 1:
            # 那么这个标签对应的文档向量就属于p1Num。
            # 则该词对应的个数p1Num就加1
            p1Num += trainMatrix[i]
            # 在所有的文档中，该文档的总词数也相应加1
            p1Denom += sum(trainMatrix[i])
        # 如果这个类型标签是非侮辱性。
        else:
            # 那么这个标签对应的文档向量就属于p0Num。
            # 则该词对应的p0Num就加1
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 最后，对每个元素除以该类别中的总词数。
    # 当计算时，由于大部分因子都非常小，所以程序会下溢出。
    # 于是通过求对数可以避免下溢出。
    p1Vect = log(p1Num/p1Denom)          #change to log()
    p0Vect = log(p0Num/p0Denom)          #change to log()
    # 返回：
    #   整体数据集中每一个单词属于侮辱性单词的概率。
    #   整体数据集中每一个单词属于非侮辱性单词的概率。
    #   类型标签的abusive概率。
    return p0Vect,p1Vect,pAbusive

# 朴素贝叶斯分类函数
# 4个输入：
#       要分类的向量vec2Classify以及
#       使用函数trainNB0()计算得到的三个概率。
#       属于P0的概率，属于P1的概率，属于侮辱性文档的概率。
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 使用NumPy的数组来计算两个向量相乘的结果。
    # 将词汇表中所有词的对应值相加，然后将该值加到类别的对数概率上。
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    # 比较类别的概率返回大概率对应的类别标签。
    if p1 > p0:
        return 1
    else: 
        return 0

# 基于词袋模型的朴素贝叶斯代码。它与函数setOfWords2Vec()几乎完全相同。
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    # 唯一不同的是，每当遇到一个单词时，增加词向量中的对应值，
    for word in inputSet:
        if word in vocabList:
            # 而不只是将对应的数值设为1。
            returnVec[vocabList.index(word)] += 1
    return returnVec

# 分类器入口函数。封装了所有的操作。
def testingNB():
    # 1. 准备数据：从文本中构建词向量。
    listOPosts,listClasses = loadDataSet()
    # 创建一个包含在所有文档中出现的不重复词的列表。
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    # 2. 使用函数setOfWords2Vec()，输入参数为词汇表及某个文档，输出文档向量。
    #    向量的每一元素为1或0，分别表示词汇表中的单词在输入文档中是否出现。
    for postinDoc in listOPosts:
        # 使用每一个词条，针对整体数据集创建整体数据集的文档向量。
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # 3. 训练算法：从词向量计算概率
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    # 4. 下面开始使用这个算法进行测试。
    testEntry = ['love', 'my', 'dalmation']
    # 给出testEntry在整体数据集中的文档向量。
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print("thisDoc : ", thisDoc)
    # 测试算法：
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print("thisDoc : ", thisDoc)
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

def textParse(bigString):    #input is big string, #output is word list
    import re
  # 这个正则表达式写错了。应该是W+取词，而不是W*取字母。
  # listOfTokens = re.split(r'\W*', bigString)
    listOfTokens = re.split(r'\W+', bigString)
    # print("listOfTokens : ", listOfTokens)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 

# 对贝叶斯垃圾邮件分类器进行自动化处理的函数。
def spamTest():
    # 这里定义了三个List。
    # 分别是用于训练的数据集List，分类标签List，
    # 和一个用于数据返回的全体数据集List。
    # 最后一个存在的原因在于全体数据集List会在后面被分为训练集和测试集。
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        # 1. 导入文件夹spam下的文本文件，并将它们解析为词列表
        # wordList = textParse(open('email/spam/%d.txt' % i).read())
        # 这里需要指定文件编码以提高通用性。
        wordList = textParse(open('email/spam/%d.txt' % i, 
            encoding='ISO-8859-1').read())        
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        # 导入文件夹ham下的文本文件，并将它们解析为词列表
        wordList = textParse(open('email/ham/%d.txt' % i, 
            encoding='ISO-8859-1').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    trainingSet = list(range(50)); testSet=[]           #create test set
    # 接下来构建一个测试集与一个训练集。
    # testSet和trainingSet中保存的是数据集docList的下标。
    for i in range(10):
        # 接下来，随机选择其中10个数字 。
        randIndex = int(random.uniform(0,len(trainingSet)))
        # 选择出的数字被添加到测试集testSet中，
        testSet.append(trainingSet[randIndex])
        # 同时也将其从训练集trainingSet中剔除。
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    # 遍历训练集的所有文档
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        # 2. 对每封邮件基于词汇表并使用词袋模型bagOfWords2VecMN()函数来构建词向量。
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    # 3. 利用词向量来计算分类所需的概率。
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    # 4. 遍历测试集，对其中每封电子邮件进行分类 。
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        # 如果邮件分类错误
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error",docList[docIndex])
    # 最后给出总的错误百分比。
    print('the error rate is: ',float(errorCount)/len(testSet))
    #return vocabList,fullText

def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True) 
    return sortedFreq[:30]       

def localWords(feed1,feed0):
    import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet=[]           #create test set
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])
