# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 17:28:40 2013

@author: attale00
"""

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def standardRF(n_estimators = 100,min_split = 10,max_depth = 20,max_features = None):
    rf = RandomForestClassifier(n_estimators=n_estimators, max_features =max_features ,max_depth=max_depth,min_samples_split=min_split, random_state=0,n_jobs=1)    
    return rf


def standardCrossvalidation(clf,dataSet,n_jobs=1,cv=5):
    scoresRF = cross_validation.cross_val_score(clf,dataSet.data,y=np.array(dataSet.targetNum),n_jobs=n_jobs,cv=cv)
    return scoresRF
    

def evaluateClassification(dataSet,mapping):
    """
    :mapping    -- a dict containing the numbers as keys with a description as value
    """
    if not dataSet.hasBeenClassified:
        print 'Only call this function with classified datasets'
        return
    
    classes = mapping.keys()
    
    nCorrect = {}
    nTotal = {}
    for k in classes:
        nCorrect[k]=0
        nTotal[k] = 0
    
    for i in xrange(len(dataSet.targetNum)):
        classifiedAs = dataSet.classifiedAs[i]
        nTotal[classifiedAs]+=1
        if classifiedAs == dataSet.targetNum[i]:
            nCorrect[classifiedAs]+=1
    
    N=len(dataSet.targetNum)
    print 'Correct Classifications({} total):'.format(N)
    for k in classes:
        msg = 'Classified correctly as '+mapping[k] +':\t{:.2} ({}/{})'.format(nCorrect[k]*1.0/nTotal[k],nCorrect[k],nTotal[k])
        n=dataSet.targetNum.count(k)
        msg1 = 'Classified {} ({}/{}) of all '.format(nCorrect[k]*1.0/n,nCorrect[k],n)+mapping[k]
        print msg
        print msg1