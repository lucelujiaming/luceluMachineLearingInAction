# coding: utf-8
'''
Created on Jun 14, 2011
FP-Growth FP means frequent pattern
the FP-Growth algorithm needs: 
1. FP-tree (class treeNode)
2. header table (use dict)

This finds frequent itemsets similar to apriori but does not 
find association rules.  

@author: Peter
'''
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        # 节点名字的变量
        self.name = nameValue
        # 节点计数值
        self.count = numOccur
        # 用于链接相似的元素项
        self.nodeLink = None
        # 指向当前节点的父节点
        self.parent = parentNode      #needs to be updated
        # 一个空字典变量，用于存放节点的子节点。
        self.children = {} 
    # 增加计数。
    def inc(self, numOccur):
        self.count += numOccur
    # 打印自己和子节点的名字和计数值。
    def disp(self, ind=1):
        # 打印自己的名字和计数值。
        print('  '*ind, self.name, ' ', self.count)
        # 打印自子节点的名字和计数值。
        for child in self.children.values():
            child.disp(ind+1)

# 这个函数的逻辑是这样的。
# 首先遍历一遍数据集。统计出来每个元素项出现的频度。放入headerTable。
# 之后过滤掉headerTable中那些些出现次数少于minSup的项。得到频繁集。
# 之后根据上面的成果和原始数据，生成FP树和更新headerTable。
# 这个headerTable是一个字典。Key是满足minSup的项的单个元素项。
# value包括两个值，一个是这个元素项出现次数。另一个是一个单向列表。
# 这个单项列表就是前面提到的相似项之间的链接即节点链接。
# 生成FP树的方法如下：
#     首先遍历数据集。找到包含的频繁集的一条数据。这里的每一条数据都包含好几个元素。
#     我们只记录满足出现次数多于minSup的元素项。
#     之后我们把记录下来的元素项按照这个元素项出现的次数排序。
#     通过把一个元素一个元素添加到子节点上面的方式，生成FP树。
#     因此上，出现次数多的更靠近根。
#     在这个FP树的构建过程中，每增加一个元素我们就需要同步更新headerTable的单向列表。
#     这个列表串联起来了相同的元素。
# 使用数据集以及最小支持度作为参数来构建FP树。
# 两个参数，
#     1. 数据集。
#     2. 最小支持度。
# Create FP-tree from dataset but don't mine
def createTree(dataSet, minSup=3): # minSup=1): 
    headerTable = {}
    # Go over dataSet twice
    # 树构建过程中会遍历数据集两次。
    # headerTable的第一阶段：
    #    第一次遍历扫描数据集并统计每个元素项出现的频度。这些信息被存储在头指针表中。
    for trans in dataSet:#first pass counts frequency of occurance
        for item in trans:
            # 统计每个元素项出现的频度。
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    print("1 --- headerTable : ", headerTable)
    # headerTable的第二阶段：
    #   扫描头指针表删掉那些出现次数少于minSup的项。
    #remove items not meeting minSup
    # for k in headerTable.keys(): 
    for k in list(headerTable.keys()):  
        if headerTable[k] < minSup: 
            # print("headerTable[", k, "] : ", headerTable[k])
            del(headerTable[k])
    # 程序执行到这里，出现次数少于minSup的项已经被删除。只剩下多于minSup的项。
    print("2 --- headerTable : ", headerTable)
    freqItemSet = set(headerTable.keys())
    # 如果所有项都不频繁，就不需要进行下一步处理。
    if len(freqItemSet) == 0: 
        return None, None  #if no items meet min support -->get out
    # headerTable的第三阶段：
    #    对头指针表稍加扩展，以便可以保存计数值及指向每种类型第一个元素项的指针。
    #    原来的value只保存一个计数值，现在保存两个值。一个计数值一个头指针。
    #    例如： {'r': 3} --> {'r': [3, None]}
    # 这个头指针表包含相同类型元素链表的起始指针。
    for k in headerTable:
        #reformat headerTable to use Node link 
        headerTable[k] = [headerTable[k], None] 
    print('3 --- headerTable: ',headerTable)
    # 创建只包含空集合的根节点。
    retTree = treeNode('Null Set', 1, None) #create tree
    # 再一次遍历数据集，
    # 值得注意的是，这里的dataSet的每一条包括两个元素，一条frozenset数据和一个计数。
    for tranSet, count in dataSet.items():  #go through dataset 2nd time
        # print("tranSet : ", tranSet, "count : ", count)
        localD = {}
        # 这次只考虑那些频繁项。
        # 循环dataSet里面的每一条frozenset数据。
        for item in tranSet:  #put transaction items in order
            # 如果这一条frozenset数据中的一个元素属于频繁项。
            if item in freqItemSet:
                # 把这个元素和对应的频繁项的计数放入localD中。
                localD[item] = headerTable[item][0]
        # 如果上面的一条frozenset数据中包含频繁项，导致localD有了数据。
        if len(localD) > 0:
            # 把获得的数据按照排序。排序基于元素项的绝对出现频率，也就是计数值来进行。
            orderedItems = [v[0] for v in sorted(localD.items(), 
                                        key=lambda p: p[1], reverse=True)]
            print("orderedItems : ", orderedItems)
            # 然后调用updateTree()方法。
            # populate tree with ordered freq itemset
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable #return tree and header table

# 为了让FP树生长，需调用updateTree。
# 其中的输入参数为：
#    一个已经按照绝对出现频率，也就是计数值排序的频繁项集。
#    FP树。
#    满足最小支持度的元素列表。
#    这个频繁项集对应的frozenset数据的计数。
# 这个函数就是通过遍历这个已经按照绝对出现频率，也就是计数值排序的频繁项集，
# 让FP树生长，同时更新headerTable的链表节点元素。
def updateTree(items, inTree, headerTable, count):
    # 首先测试事务中的第一个元素项是否作为子节点存在。
    # 因为这个函数会被一层层迭代调用进去。早晚会遇到这种情况。当然一开始肯定不是这样的。
    # 如果存在的话，
    if items[0] in inTree.children:#check if orderedItems[0] in retTree.children
        # 则更新该元素项的计数；
        # 更新方法是增加元素所在的频繁项集对应的frozenset数据的计数。
        inTree.children[items[0]].inc(count) #incrament count
    # 如果不存在，
    else:   #add items[0] to inTree.children
        # 则创建一个新的treeNode并将其作为一个子节点添加到树中。
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        # 头指针表也要更新以指向新的节点。
        # 如果元素列表中items[0]对应的项的头指针没有被设置过，为空。
        if headerTable[items[0]][1] == None: #update header table 
            # 第一次指向自己。
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            # 否则如果之前设置过，则需要更新头指针表。
            # 这个头指针表包含相同类型元素链表的起始指针。
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    # 接着不断迭代调用自身，每次调用时会去掉列表中第一个元素。
    if len(items) > 1:#call updateTree() with remaining ordered items
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

# 用于确保节点链接指向树中该元素项的每一个实例。
# this version does not use recursion
def updateHeader(nodeToTest, targetNode):   
    # 从头指针表的nodeLink开始，一直沿着nodeLink直到到达链表末尾。
    # Do not use recursion to traverse a linked list!
    while (nodeToTest.nodeLink != None):   
        nodeToTest = nodeToTest.nodeLink
    # 前面创建出来FP树的新节点加到头指针链表的尾部。
    nodeToTest.nodeLink = targetNode

# 有了FP树之后，就可以抽取频繁项集了。
# 循环上溯FP树，
def ascendTree(leafNode, prefixPath): #ascends from leaf node to root
    if leafNode.parent != None:
        # 收集所有遇到的元素项的名称
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)
# 查找以所查找元素项为结尾的路径集合。
# 包括两个参数：
#    第一个参数：元素项的名字。没有使用。
#    第二个参数：元素项。
# 如果我们明白前面的headerTable的含义。方法就非常简单了。
# 首先headerTable这个字典中，每一个元素项都有一个链表。连接了FP树中所有的相同的元素项。
# 因此我们只需要遍历这个链表项。针对每一个链表项，就是找到一个节点。之后在FP树中上溯。
# 记录下沿途FP树中父节点的名字。就可以得到这个元素在FP树中的所有Path了。
def findPrefixPath(basePat, treeNode): #treeNode comes from header table
    # 这里使用字典的原因是，后面添加的时候，会出现大量的重复性添加。
    # 就是反反复复添加同样的内容。
    condPats = {}
    # 遍历链表直到到达结尾。
    while treeNode != None:
        prefixPath = []
        # 每遇到一个元素项都会调用ascendTree()来上溯FP树，
        # 在这个过程中，并收集所有遇到的元素项的名称。放在prefixPath中。
        ascendTree(treeNode, prefixPath)
        print("prefixPath : ", prefixPath)
        # 该列表返回之后添加到条件模式基字典condPats中。
        if len(prefixPath) > 1: 
            print("frozenset(prefixPath[1:]) : ", frozenset(prefixPath[1:]))
            print("treeNode : ", treeNode.count)
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        # 指向下一个元素。
        treeNode = treeNode.nodeLink
    return condPats

# 对于每一个频繁项，创建条件FP树的代码。
# 包括5个参数：
#    前面构建出来的FP树。这里没有用到。
#    前面返回的headerTable字典。
#    最小支持度。因为一个元素在整个FP上满足最小支持度不等于在一个频繁项的条件FP树上也满足。
#    前缀集合。后面频繁项集列表中每一个元素的前缀。
#    频繁项集列表。
# 这个函数的逻辑是这样的。首先因为headerTable保存了所有的相似元素。
# 因此上，我们基于这些相似元素挨个构建FP树。
# 并在构建过程中记录得到的频繁项集，保存在freqItemList中。
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    #(sort header table)
    # bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1])]
    # bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]
    # 程序首先对头指针表中的元素项按照其出现频率进行排序。
    #（记住这里的默认顺序是按照从小到大。）
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: str(p[1]))]
    print("bigL : ", bigL)
    # 1. start from bottom of header table
    for basePat in bigL:  
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        print('finalFrequent Item: ',newFreqSet)    #append to set
        # 将每一个频繁项添加到频繁项集列表freqItemList中。
        freqItemList.append(newFreqSet)
        # 递归调用findPrefixPath()函数来创建条件基。
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        print('condPattBases :',basePat, condPattBases)
        # 2. construct cond FP-tree from cond. pattern base
        # 该条件基被当成一个新数据集输送给createTree()函数。
        myCondTree, myHead = createTree(condPattBases, minSup)
        print('head from conditional tree: ', myHead)
        # 最后，如果树中有元素项的话，递归调用mineTree()函数。
        # 如果myHead为空，说明condPattBases的元素都不满足最小支持度，没有多于minSup的项。
        # 否则如果myHead不为空，说明condPattBases中有一些满足最小支持度的元素。
        # 而且createTree根据这些元素已经构建了FP树。
        # 那就需要让这颗构建好的FP树继续生长。因此上，需要迭代调用mineTree。
        # 3. mine cond. FP-tree
        if myHead != None: 
            print('conditional tree for: ',newFreqSet)
            myCondTree.disp(1)            
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

'''
import twitter
from time import sleep
import re

def textParse(bigString):
    urlsRemoved = re.sub('(http:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*', '', bigString)    
    listOfTokens = re.split(r'\W*', urlsRemoved)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def getLotsOfTweets(searchStr):
    CONSUMER_KEY = ''
    CONSUMER_SECRET = ''
    ACCESS_TOKEN_KEY = ''
    ACCESS_TOKEN_SECRET = ''
    api = twitter.Api(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_TOKEN_KEY, 
                      access_token_secret=ACCESS_TOKEN_SECRET)
    #you can get 1500 results 15 pages * 100 per page
    resultsPages = []
    for i in range(1,15):
        print("fetching page %d" % i)
        searchResults = api.GetSearch(searchStr, per_page=100, page=i)
        resultsPages.append(searchResults)
        sleep(6)
    return resultsPages

def mineTweets(tweetArr, minSup=5):
    parsedList = []
    for i in range(14):
        for j in range(100):
            parsedList.append(textParse(tweetArr[i][j].text))
    initSet = createInitSet(parsedList)
    myFPtree, myHeaderTab = createTree(initSet, minSup)
    myFreqList = []
    mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
    return myFreqList
'''
#minSup = 3
#simpDat = loadSimpDat()
#initSet = createInitSet(simpDat)
#myFPtree, myHeaderTab = createTree(initSet, minSup)
#myFPtree.disp()
#myFreqList = []
#mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
