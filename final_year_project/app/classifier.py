
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import collections
import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
import pickle
import requests,json


# In[2]:


data=pd.read_csv("data.csv",delimiter=None)

def bag_of_words(words):
    return dict([(word,True) for word in words])

def label_feats_extractor(data,feature_detector=bag_of_words):
    label_feats = collections.defaultdict(list)
    for i in range(1000):
        feats=feature_detector(data['Review'][i].split())
        label=data['Category'][i]
        label_feats[label].append(feats)
    return label_feats

def split_label_feats(lfeats, split=0.75):
    train_feats = []
    test_feats = []
    for label,feats in lfeats.items():
        cutoff = int(len(feats)*split)
        train_feats.extend([(feat,label) for feat in feats[:cutoff]])
        test_feats.extend([(feat,label) for feat in feats[cutoff:]])
    return train_feats,test_feats

lfeats=label_feats_extractor(data)
train_feats, test_feats = split_label_feats(lfeats, split=0.75)

nb_classifier = NaiveBayesClassifier.train(train_feats)
#print(nltk.classify.accuracy(nb_classifier, test_feats))
#nb_classifier.show_most_informative_features(10)
pickle.dump(nb_classifier,open("nb_class.pkl","wb"))
