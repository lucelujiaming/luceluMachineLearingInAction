-----------------------generateRules---------------------------------

calcConf deals H1 :  [frozenset({2}), frozenset({3})]
supportData[ frozenset({2, 3}) ] =  0.5  supportData[ frozenset({3}) ] =  0.75 conf = 0.6666666666666666
frozenset({3}) --> frozenset({2}) conf: 0.6666666666666666
supportData[ frozenset({2, 3}) ] =  0.5  supportData[ frozenset({2}) ] =  0.75 conf = 0.6666666666666666
frozenset({2}) --> frozenset({3}) conf: 0.6666666666666666
calcConf deals H1 :  [frozenset({3}), frozenset({5})]
supportData[ frozenset({3, 5}) ] =  0.5  supportData[ frozenset({5}) ] =  0.75 conf = 0.6666666666666666
frozenset({5}) --> frozenset({3}) conf: 0.6666666666666666
supportData[ frozenset({3, 5}) ] =  0.5  supportData[ frozenset({3}) ] =  0.75 conf = 0.6666666666666666
frozenset({3}) --> frozenset({5}) conf: 0.6666666666666666
calcConf deals H1 :  [frozenset({2}), frozenset({5})]
supportData[ frozenset({2, 5}) ] =  0.75  supportData[ frozenset({5}) ] =  0.75 conf = 1.0
frozenset({5}) --> frozenset({2}) conf: 1.0
supportData[ frozenset({2, 5}) ] =  0.75  supportData[ frozenset({2}) ] =  0.75 conf = 1.0
frozenset({2}) --> frozenset({5}) conf: 1.0
calcConf deals H1 :  [frozenset({1}), frozenset({3})]
supportData[ frozenset({1, 3}) ] =  0.5  supportData[ frozenset({3}) ] =  0.75 conf = 0.6666666666666666
frozenset({3}) --> frozenset({1}) conf: 0.6666666666666666
supportData[ frozenset({1, 3}) ] =  0.5  supportData[ frozenset({1}) ] =  0.5 conf = 1.0
frozenset({1}) --> frozenset({3}) conf: 1.0
rulesFromConseq deals H1 :  [frozenset({2}), frozenset({3}), frozenset({5})]
rulesFromConseq -> calcConf deals H1 :  [frozenset({2}), frozenset({3}), frozenset({5})]
supportData[ frozenset({2, 3, 5}) ] =  0.5  supportData[ frozenset({5}) ] =  0.75 conf = 0.6666666666666666
frozenset({5}) --> frozenset({2, 3}) conf: 0.6666666666666666
supportData[ frozenset({2, 3, 5}) ] =  0.5  supportData[ frozenset({3}) ] =  0.75 conf = 0.6666666666666666
frozenset({3}) --> frozenset({2, 5}) conf: 0.6666666666666666
supportData[ frozenset({2, 3, 5}) ] =  0.5  supportData[ frozenset({2}) ] =  0.75 conf = 0.6666666666666666
frozenset({2}) --> frozenset({3, 5}) conf: 0.6666666666666666
len(Hmp1) > 1 and Hmp1 =  [frozenset({2, 3}), frozenset({2, 5}), frozenset({3, 5})]
rules :
 [(frozenset({3}), frozenset({2}), 0.6666666666666666), 
  (frozenset({2}), frozenset({3}), 0.6666666666666666), 
  (frozenset({5}), frozenset({3}), 0.6666666666666666), 
  (frozenset({3}), frozenset({5}), 0.6666666666666666), 
  (frozenset({5}), frozenset({2}), 1.0), 
  (frozenset({2}), frozenset({5}), 1.0), 
  (frozenset({3}), frozenset({1}), 0.6666666666666666), 
  (frozenset({1}), frozenset({3}), 1.0), 
  (frozenset({5}), frozenset({2, 3}), 0.6666666666666666), 
  (frozenset({3}), frozenset({2, 5}), 0.6666666666666666), 
  (frozenset({2}), frozenset({3, 5}), 0.6666666666666666)]
(base) lucedeMacBook-Pro:Ch11 lucelu$
