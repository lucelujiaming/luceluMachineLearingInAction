# coding: utf-8

import re
import bayes
import matplotlib
from matplotlib import pyplot as plt

myString = 'This book is the best book on Python or ML. I have \
ever laid eyes upon.'
# print("myString : ", myString.split())

# listOfTokens = bayes.textParse(myString)
regEx = re.compile(r'\W+')
listOfWords = regEx.split(myString)
listOfTokens = [tok.lower() for tok in listOfWords if len(tok) > 2] 
# print("listOfTokens : ", listOfTokens)

# emailText = open('email/ham/6.txt', encoding='ISO-8859-1').read()
# listOfTokens = regEx.split(emailText)
# print("listOfTokens : ", listOfTokens)

bayes.spamTest()
