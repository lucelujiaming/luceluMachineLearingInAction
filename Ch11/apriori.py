# coding: utf-8
'''
Created on Mar 24, 2011
Ch 11 code
@author: Peter
'''
from numpy import *

# Apriori算法的思想是这样的。
# 假设一家上海的连锁超市，有1000种产品。每个月有100万个顾客光顾。每个顾客买5到20种商品。
# 现在超市想找出来，一些畅销的产品组合，用来促销。
# 最笨的方法当然就是把用户购买过的商品进行两两组合，三三组合，四四组合。之后遍历这100万条交易记录，
# 针对这些用户购买过的商品组合，在100万条交易记录中查找出现次数。
# 这种方法当然是不可接受的。
# 为了不用愚蠢的一遍一遍的遍历整个数据集，因此上我们引入了Apriori原理。
# 就是对于一个畅销的产品组合，其子集是必然是畅销的。
# 反之，对于一个滞销的产品组合，其子集是必然是滞销的。
# 因此上，Apriori算法的步骤如下：
# 我们首先得到所有产品的组合作为C1。之后在交易记录中查找得到单个产品的畅销组合L1，
# 当然这里的组合只有一个产品。之后把L1中的产品两两组合加入C2。
# 之后在交易记录中查找得到两个产品的畅销组合L2。
# 以此类推即可。

# 有了这些畅销组合，我们需要建立这些畅销组合的关系。也就是从频繁项集中挖掘关联规则。
# 因为一个畅销的产品组合，其子集是必然是畅销的。为了减少关联规则数量，我们引入了可信度。
# 一条规则P ➞ H的可信度定义为support(P | H)/support(P)。
# 为了实现采用可信度过滤畅销组合和他们的子集。

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
# 构建集合C1。C1是大小为1的所有候选项集的集合。
def createC1(dataSet):
    # 首先创建一个空列表C1，它用来存储所有不重复的项值。
    C1 = []
    # 遍历数据集中的所有交易记录。
    for transaction in dataSet:
        # 对每一条记录，遍历记录中的每一个项。
        for item in transaction:
            # 如果某个物品项没有在C1中出现，
            if not [item] in C1:
                # 则将其添加到C1中。
                # 不是简单地添加每个物品项，而是添加只包含该物品项的一个列表。
                # C1是一个集合的集合。
                C1.append([item])
    # 对大列表进行排序
    C1.sort()
    # 并将其中的每个单元素列表映射到frozenset()，
    # 最后返回frozenset的列表
    # 使用的格式就是Python中的frozenset类型。
    # map只能循环一次。因此上需要换成list
    return list(map(frozenset, C1))#use frozen set so we
                            #can use it as a key in a dict    

# 数据集扫描的伪代码大致如下：
#   对数据集中的每条交易记录tran
#   对每个候选项集can：
#       检查一下can是否是tran的子集：
#       如果是，则增加can的计数值
#   对每个候选项集：
#   如果其支持度不低于最小值，则保留该项集
#   返回所有频繁项集列表
# 三个参数，
#     1. 集合表示的数据集。
#     2. 包含候选集合的列表。也就是前面createC1的返回值。
#        类型为frozenset的集合。
#     3. 感兴趣项集的最小支持度minSupport。
def scanD(D, Ck, minSupport):
    # 首先创建一个空字典ssCnt，key是C1中的每一条frozenset条目，
    # value是C1中的这一条frozenset条目在数据集中出现的次数。
    ssCnt = {}
    # print("lucelu D : \n", D)
    # 遍历数据集中的所有交易记录。
    for tid in D:
        # 遍历C1中的每一条frozenset条目。
        for can in Ck:
            # 如果当前的这一条frozenset条目是当前交易记录的一部分，
            if can.issubset(tid):
                # 那么增加字典中对应的计数值。
                # if not ssCnt.has_key(can): 
                if can in ssCnt:
                    ssCnt[can] += 1
                else: 
                    ssCnt[can]=1
    # print("ssCnt : \n", ssCnt)
    # 当扫描完数据集中的所有项以及所有候选集时，就需要计算支持度。
    # 首先计算得到数据集中的交易记录的条数。
    numItems = float(len(D))
    # print("numItems : ", numItems)
    # 会先构建一个空列表，该列表包含满足最小支持度要求的集合。
    retList = []
    supportData = {}
    # 遍历字典ssCnt中的每个元素并且计算支持度。
    for key in ssCnt:
        # 支持度计算方法就是一条frozenset条目在数据集中出现的次数处以数据集总数。
        support = ssCnt[key]/numItems
        # print("ssCnt[key] : ", ssCnt[key], " and support : ", support)
        # 如果支持度满足最小支持度要求，则将字典元素添加到retList中。
        # 这样那些不满足最小支持度的集合会被去掉。
        if support >= minSupport:
            retList.insert(0,key)
        # 这个supportData字典。key是C1中的每一条frozenset条目，
        # value是C1中的这一条frozenset条目在数据集中出现的次数处以数据集总数。
        # 也就是支持度。
        supportData[key] = support
    # 函数最后返回满足最小支持度要求的集合retList和最频繁项集的支持度supportData。
    return retList, supportData

# 创建候选项集Ck
# 输入参数为频繁项集列表Lk与项集元素个数k
def aprioriGen(Lk, k): #creates Ck
    # 创建一个空列表，
    retList = []
    # 然后计算Lk中的元素数目。
    lenLk = len(Lk)
    # 通过两个for循环来比较Lk中的每一个元素与其他元素，
    for i in range(lenLk):
        for j in range(i+1, lenLk): 
            # 取列表中的两个集合进行比较。比较前面k-2个元素
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            # 如果前面k-2个元素都相等，
            # 这里使用了一个技巧来避免遍历列表元素。
            if L1==L2: #if first k-2 elements are equal
                # 那么就将这两个集合合成一个大小为k的集合
                retList.append(Lk[i] | Lk[j]) #set union
    return retList

# 这个Apriori算法函数的流程是这样的。
#     首先调用createC1生成C1。
#     对于数据[{1, 3, 4}, {2, 3, 5}, {1, 2, 3, 5}, {2, 5}]来说，
#     一共有4条数据。
#     生成的C1肯定就是：[1, 2, 3, 4, 5]
#     之后调用scanD(DD, C1, 0.5)得到L1。
#     低于1来说，在4条数据中出现的次数为2次，则支持度就为50%
#     因为这里指定的最小支持度为0.5。
#     那就意味着只有出现概率在50%上的元素才能计入L1。
#     因此上L1的数据就是[1, 2, 3, 5]。4因为只出现一次，无法进入下一轮。
#     下面就是调用aprioriGen从L1得到C2
#     C2就是当前L1的两两组合。
#     也就是{{1, 2}, {1, 3}), {1, 5}, {2, 3}, {2, 5}): {3, 5}}
#     之后我们调用scanD(DD, C2, 0.5)得到L2。
#     因为这里指定的最小支持度为0.5。
#     那就意味着只有出现概率在50%上的元素才能计入L2。
#     以此反复。直到L*变成空集合。
#     最后我们就返回生成的这个L1...L*的频繁项列表。
#     同时返回这个列表中每一个元素的支持度。
# 整个Apriori算法的伪代码如下：
#   当集合中项的个数大于0时
#       构建一个k个项组成的候选项集的列表
#       检查数据以确认每个项集都是频繁的
#       保留频繁项集并构建k+1项组成的候选项集的列表
# 参数为一个数据集以及一个支持度。
def apriori(dataSet, minSupport = 0.5):
    # 首先生成候选项集的列表。首先创建C1然后读入数据集将其转化为D。
    C1 = createC1(dataSet)
    # 使用map函数将set()映射到dataSet列表中的每一项。
    D = list(map(set, dataSet))
    # 使用程序scanD()函数来创建L1，
    L1, supportData = scanD(D, C1, minSupport)
    # 将L1放入列表L中。
    L = [L1]
    k = 2
    # 现在有了L1，后面会继续找L2，L3…，工作流程如下：
    while (len(L[k-2]) > 0):
        # 首先使用aprioriGen()来创建Ck，
        Ck = aprioriGen(L[k-2], k)
        # 然后使用scanD()基于Ck来创建Lk。Ck是一个候选项集列表，
        # 然后scanD()会遍历Ck，丢掉不满足最小支持度要求的项集。
        Lk, supK = scanD(D, Ck, minSupport)#scan DB to get Lk
        supportData.update(supK)
        # Lk列表被添加到L，同时增加k的值，
        L.append(Lk)
        k += 1
    # 重复上述过程。当Lk为空时，程序返回L并退出。
    # print("L : ", L)
    return L, supportData

# 有了L1...L*的频繁项列表和这个列表中每一个元素的支持度，
# 我们就可以生成关联规则了。一共包括三个函数。
# 主函数的工作就是遍历频繁项集列表的每一层中的每一个频繁项。
# 值得注意的是在遍历的过程中，频繁项会在calcConf和rulesFromConseq中被传递。而不会被修改。
# 使用列表推导获得这个频繁项中的每一个元素。
# 之后就分成两种情况。第一种情况是：
#    当频繁项频繁项中的元素只有两个元素的时候，调用calcConf()来计算可信度值。
#    把满足最小可信度阈值的规则加入bigRuleList列表。
#    当然，这个时候，生成的规则要么没有要么就是{A --> B}和{B --> A}
# 第二种情况是：
#    当频繁项频繁项中的元素多于两个元素的时候，首先调用aprioriGen生成所有可能的规则。
#    之后调用calcConf()来计算可信度值。把满足最小可信度阈值的规则加入bigRuleList列表。
#    之后，如果发现bigRuleList列表新增的规则超过1条。
#    说明当前频繁项有多条规则到达前面调用aprioriGen生成的规则。

# 关联规则生成的主函数
# 有3个参数：
#       频繁项集列表。
#       包含那些频繁项集支持数据的字典、
#       最小可信度阈值。默认值设为0.7
# 前两个输入参数正好是程序清单11-2中函数apriori()的输出结果。
# 返回一个包含可信度的规则列表。
# supportData is a dict coming from scanD
def generateRules(L, supportData, minConf=0.7):  
    # 一个包含可信度的规则列表。
    # 这个列表中每一个元素包含三个部分，规则的起点元素，规则的终点元素，可信度。
    bigRuleList = []
    # 遍历L中的每一个频繁项集
    # 这个频繁项集长成这个样子。
    #  [[{5}, {2}, {3}, {1}], [{2, 3}, {3, 5}, {2, 5}, {1, 3}], [{2, 3, 5}], []]
    # 可以看出来，整个频繁项集分为三个阶层。
    # 第一层是L1, L2, L3 .....，也就是下面的外层循环。
    # 第二层是每一个L*中，都包含了若干个元素。也就是下面的内层循环。
    # 例如L2的内容就是这样： [{2, 3}, {3, 5}, {2, 5}, {1, 3}]
    # 第三层是L*层中，每一个元素都包含了一个或多个数据。也就是列表推导的部分。
    # print("L : ", L)
    # 这里从Index = 1的第二层，也就是L2开始遍历。
    for i in range(1, len(L)):#only get the sets with two or more items
        # print("L[", i, "] : ", L[i])
        # 遍历当前频繁项中的每一个元素。
        for freqSet in L[i]:
            # print("freqSet : ", freqSet)
            # 对每个频繁项集创建只包含单个元素集合的列表H1。
            H1 = [frozenset([item]) for item in freqSet]
            # 如果频繁项集的元素数目超过2，也就是外层循环到了第2轮，开始处理L3以后，进入这个分支。
            # 就可以考虑对它做进一步的合并。
            if (i > 1):
                # 进行具体合并。
                # freqSet是当前的包含多个数据的一个元素。例如：{2, 5}
                # H1是freqSet中元素的列表。
                # supportData记录了freqSet和H1这些频繁项的支持度。
                print("rulesFromConseq deals H1 : ", H1)
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            # 如果项集中只有两个元素，那么使用函数calcConf()来计算可信度值。
            else:
                # freqSet是当前的包含多个数据的一个元素。例如：{2, 5}
                # H1是freqSet中元素的列表。
                # supportData记录了freqSet和H1这些频繁项的支持度。
                print("calcConf deals H1 : ", H1)
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList         
# 对规则进行评估
# 计算可信度值以及找到满足最小可信度要求的规则。
# 这个函数的思路就是，为找到感兴趣的规则，我们先调用aprioriGen生成一个可能的规则列表，
# 然后在calcConf里面测试每条规则的可信度。如果可信度不满足最小要求，则去掉该规则。
# 关于prunedH。它返回的是传入的H中满足最小可信度的元素组合。
#   比方说当从generateRules传入的时候，H只有两个元素，但是这种情况下，返回值会被忽略。
#   因为虽然有两个元素，遍历这两个单独元素的工作在calcConf已经完成了。不会出现遗漏。
#   而当从rulesFromConseq传入的时候，H是前一级元素两两组合的结果。这种情况下，返回值就有用了。
#   这个时候，需要迭代调用rulesFromConseq进一步遍历返回的满足最小可信度的元素组合。
#   否则就会发生元素组合遍历上的遗漏。
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    # 创建一个空列表。作为满足最小可信度要求的规则列表。
    prunedH = [] #create new list to return
    # 遍历H中的所有项集。
    # 也就是针对每一个数据点进行遍历。
    for conseq in H:
        # 计算它们的可信度值。
        # 一条规则P ➞ H的可信度定义为support(P | H)/support(P)。@P209
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calc confidence
        print("supportData[", freqSet, "] = ", supportData[freqSet], \
              " supportData[", freqSet-conseq, "] = ", supportData[freqSet - conseq], "conf =", conf)
        # 如果某条规则满足最小可信度值，
        if conf >= minConf: 
            # 那么将这些规则输出到屏幕显示。通过检查的规则也会被返回。
            # 打印< P --> H >
            print(freqSet-conseq,'-->',conseq,'conf:',conf)
            # brl是前面通过检查的bigRuleList。
            # 每一个元素包含三个部分，规则的起点元素，规则的终点元素，可信度。
            # 表示一条从不包含conseq元素的集合指向conseq元素的规则。且支持度为conf。
            brl.append((freqSet-conseq, conseq, conf))
            # 把conseq元素加入prunedH。
            prunedH.append(conseq)
    return prunedH

# 生成候选规则集合
# 为从最初的项集中生成更多的关联规则。
# 有2个参数：
#       一个是频繁项集，
#       另一个是可以出现在规则右部的元素列表H。
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    # 先计算H中的频繁集大小m。
    m = len(H[0])
    # 查看该频繁项集是否大到可以移除大小为m的子集。
    if (len(freqSet) > (m + 1)): #try further merging
        # 使用函数aprioriGen()来生成H中元素的无重复组合。
        # 该结果会存储在Hmp1中，这也是下一次迭代的H列表。
        Hmp1 = aprioriGen(H, m+1)#create Hm+1 new candidates
        # Hmp1包含所有可能的规则。
        # 利用calcConf()根据Hmp1的每一个元素集，使用频繁项集中这个元素集对应的支持度，
        # 来测试它们的可信度以确定规则是否满足要求。
        # 返回calcConf发现的新规则的终点元素。
        print("rulesFromConseq -> calcConf deals H1   : ", H)
        print("rulesFromConseq -> calcConf deals Hmp1 : ", Hmp1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        print("rulesFromConseq -> calcConf return     : ", Hmp1)
        # 如果前面的calcConf发现了多条规则，
        # 也就是说，上面通过aprioriGen生成的无重复组合，在频繁项集中，有不止一条规则满足可信度要求，
        # 说明在这个aprioriGen生成的无重复组合上面，存在着更大的组合，在频繁项集中，也可以满足可信度要求。
        # 比方说，如果{1,3,7} --> {3}, {1,3,7} --> {7}, 那么必然存在{1,3,7} --> {3,7}
        # 这里再一次运用了Apriori原理。
        if (len(Hmp1) > 1):    #need at least two sets to merge
            # 迭代调用函数rulesFromConseq()来判断是否可以进一步组合这些规则。
            print("len(Hmp1) > 1 and Hmp1 =             ", Hmp1)
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

def pntRules(ruleList, itemMeaning):
    for ruleTup in ruleList:
        for item in ruleTup[0]:
            print(itemMeaning[item])
        print("           -------->")
        for item in ruleTup[1]:
            print(itemMeaning[item])
        print("confidence: %f" % ruleTup[2])
        # print(      #print(a blank line
        
          
from time import sleep
from votesmart import votesmart
votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
#votesmart.apikey = 'get your api key first'
# 返回存储在recent20bills.txt文件中议案的actionId。
def getActionIds():
    # 创建两个空列表。这两个列表分别用来返回actionsId和标题。
    actionIdList = []; billTitleList = []
    # 打开recent20bills.txt文件，对每一行内不同元素使用tab进行分隔
    fr = open('recent20bills.txt') 
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.votes.getBill(billNum) #api call
            for action in billDetail.actions:
                if action.level == 'House' and \
                (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print('bill: %d has actionId: %d' % (billNum, actionId))
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print("problem getting bill %d" % billNum)
        sleep(1)                                      #delay to be polite
    return actionIdList, billTitleList
        
def getTransList(actionIdList, billTitleList): #this will return a list of lists containing ints
    itemMeaning = ['Republican', 'Democratic']#list of what each item stands for
    for billTitle in billTitleList:#fill up itemMeaning list
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}#list of items in each transaction (politician) 
    voteCount = 2
    for actionId in actionIdList:
        sleep(3)
        print('getting votes for actionId: %d' % actionId)
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candidateName): 
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except: 
            print("problem getting actionId: %d" % actionId)
        voteCount += 2
    return transDict, itemMeaning
