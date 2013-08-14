# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 17:28:40 2013

@author: attale00
"""

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import PyRoc
import pickle
import plottingUtils
import matplotlib.pyplot as plt

def standardRF(n_estimators = 100,min_split = 10,max_depth = 20,max_features = None,min_samples_leaf=1):
    rf = RandomForestClassifier(n_estimators=n_estimators, max_features =max_features ,max_depth=max_depth,min_samples_split=min_split, random_state=0,n_jobs=1,min_samples_leaf=min_samples_leaf)    
    return rf


def standardCrossvalidation(clf,dataSet,n_jobs=1,cv=5):
    scoresRF = cross_validation.cross_val_score(clf,dataSet.data,y=np.array(dataSet.targetNum),n_jobs=n_jobs,cv=cv)
    return scoresRF

def dissectedCV(clf,dataSet,cv=3,mapping={0:'closed or narrow',1:'open or wide open'}):
    truePositiveRate=[]
    trueNegativeRate = []
    accuracy=[]
    rocList = []
    rocPlotLabels = []
    for i in range(cv):
        training, test = dataSet.splitInTestAndTraining(frac=(1-1./cv))
        clf.fit(training.data,training.targetNum)
        test.classifiedAs=clf.predict(test.data)
        test.probabilities = clf.predict_proba(test.data)
        rocList.append(ROCCurve(test))
        
        test.hasBeenClassified = True
        tpr,tnr,acc=evaluateClassification(test,mapping)
        truePositiveRate.append(tpr)
        trueNegativeRate.append(tnr)
        accuracy.append(acc)
        rocPlotLabels.append('Fold # {}'.format(i+1))
    print 'Summary: '
    msg='TPR: {} \nTNR: {} \nAccuracy: {}'.format(np.mean(truePositiveRate),np.mean(trueNegativeRate),np.mean(accuracy))
    print msg
    PyRoc.plot_multiple_roc(rocList,include_baseline=True,labels=rocPlotLabels,plot_average=False)
    #plt.show()
    return PyRoc.multiple_roc_average(rocList,binstep=.05)    
    
def splitByPose(testSet):
    p=['140', '080', '190', '130', '050', '051', '041']
    poses={}    
    for i in p:
        poses[i]=[0,0]
        
    for i in range(len(testSet.fileNames)):
        pose = testSet.fileNames[i][10:13]
        if testSet.targetNum[i] != testSet.classifiedAs[i]:
            #error            
            poses[pose][0]+=1
        else:
            poses[pose][1]+=1
    return poses

def splitByPoseFull(testSet):
  
    p={'140':'15', '080':'45', '190':'45', '130':'30', '050':'15', '051':'0', '041':'30'}
    poses={}    
    for i in p.values():
        poses[i]=[]
        
    for i in range(len(testSet.fileNames)):
        pose = p[testSet.fileNames[i][10:13]]
        poses[pose].append((testSet.targetNum[i],testSet.probabilities[i,1]))
    return poses
    

def ROCCurve(dataSet):
    t=zip(dataSet.targetNum,dataSet.probabilities[:,1])
    rocObj = PyRoc.ROCData(t,linestyle='r-')
    return rocObj

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
    if len(mapping.keys())==2:
        n=dataSet.targetNum.count(0)
        msg='True Positive Rate (Sensitivity): {} \n True Negative Rate (Specificity): {}'.format(nCorrect[1]*1./(N-n),nCorrect[0]*1./n)
        print msg
        return (nCorrect[1]*1./(N-n),nCorrect[0]*1./n,(nCorrect[1]+nCorrect[0])*1./N)
        
def classifyWithOld(path,testSet,mode,clfPath,viewPath=None,thresh =.5):
    #f=file('/home/attale00/Desktop/classifiers/RandomForestMouthclassifier_1','r')
    f=file(clfPath,'r')
    print 'classifier used: '+ f.name
    clf = pickle.load(f)
    testSet.classifiedAs=clf.predict(testSet.data)
    testSet.probabilities=clf.predict_proba(testSet.data)
    rocObj = ROCCurve(testSet)
    for i in range(len(testSet.data)):
        if testSet.probabilities[i][1]>=thresh:
            testSet.classifiedAs[i]=1
        else:
            testSet.classifiedAs[i]=0
    testSet.hasBeenClassified = True
    pdic = splitByPose(testSet)
    #pickle.dump(pdic,open('/home/attale00/Desktop/patchPose','w'))
    if mode =='s':
        _score(clf,testSet)
    else:
        _view(clf,testSet,viewPath)
        _score(clf,testSet)
    #plottingUtils.plotPoses(pdic)
        
    rocObj.plot(include_baseline=True)
    return rocObj

def _score(clf,testSet):
    score = clf.score(testSet.data,testSet.targetNum)
    testSet.hasBeenClassified = True
    evaluateClassification(testSet,{0:'closed or narrow',1:'open or wide open'})
    print 'Overall Score: {:.3f}'.format(score)
    return
def _view(clf,testSet,path):
    viewer = plottingUtils.ClassifiedImViewer(path,testSet)
    viewer.view(comparer=plottingUtils.MouthTwo2FourComparer)
    