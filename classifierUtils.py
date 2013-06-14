# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 17:28:40 2013

@author: attale00
"""

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def standardRF(n_estimators = 100,min_split = 10,max_depth = 20,max_features = None,min_samples_leaf=1):
    rf = RandomForestClassifier(n_estimators=n_estimators, max_features =max_features ,max_depth=max_depth,min_samples_split=min_split, random_state=0,n_jobs=1,min_samples_leaf=min_samples_leaf)    
    return rf


def standardCrossvalidation(clf,dataSet,n_jobs=1,cv=5):
    scoresRF = cross_validation.cross_val_score(clf,dataSet.data,y=np.array(dataSet.targetNum),n_jobs=n_jobs,cv=cv)
    return scoresRF

def dissectedCV(clf,dataSet,cv=3,mapping={0:'closed or narrow',1:'open or wide open'}):
    for i in range(cv):
        training, test = dataSet.splitInTestAndTraining(frac=(1-1./cv))
        clf.fit(training.data,training.targetNum)
        test.classifiedAs=clf.predict(test.data)
        test.hasBeenClassified = True
        evaluateClassification(test,mapping)

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
        if nTotal[k]==0:
            print 'adding one to nTotaal'
            nTotal[k]+=1
        msg = 'Classified correctly as '+mapping[k] +':\t{:.2} ({}/{})'.format(nCorrect[k]*1.0/nTotal[k],nCorrect[k],nTotal[k])
        n=dataSet.targetNum.count(k)
        if n == 0:
            print 'adding one to n'
            n+=1
        msg1 = 'Classified {} ({}/{}) of all '.format(nCorrect[k]*1.0/n,nCorrect[k],n)+mapping[k]
        print msg
        print msg1