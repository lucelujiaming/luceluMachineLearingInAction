# coding: utf-8

import re
import bayes
import feedparser

ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
# Your request has been blocked.
print("ny['entries']", ny['entries'])