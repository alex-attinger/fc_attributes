# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:35:54 2013

@author: attale00
"""

## mouth

import utils
import numpy as np
import featureGeneration as fg
from sklearn import svm, cross_validation
from sklearn.ensemble import RandomForestClassifier
import sys
import plottingUtils
import matplotlib.pyplot as plt

def main(nJobs = 1):

    path = '/local/attale00/GoodPose'
    path_ea = path+'/extracted_alpha'
    path_adata = path_ea + '/a_data'
    
    filenames = utils.getAllFiles(path+'/targets');
    
    attribute = 'mouth'
    
    attribute_values = utils.parseLabelINIFile(path+'/mouth_labels/labels.ini',attribute);
    
    print('------------Attribute: \t'+attribute+' ---------------')
    for i in attribute_values:
        print('Value: \t'+i)
        
    print('----------------------------')
    print('----------parsing label files------')
    labs=utils.parseLabelFiles(path+'/mouth_labels','mouth',filenames,cutoffSeq='.png',suffix='_face0.labels')
    print('-----computing Features-----')
    #make 10 bin hist for each mouth
    mouthSet = fg.getHistogram(10,(0,270,60,452),hrange=(155.,255.0),labelFileDict = labs,path = path_ea+'/grayScale/',ending='_0.png')
    
    mouthSet.targetNum = map(utils.mapMouthLabels2Two,mouthSet.target)
    fpr=[]
    tpr=[]
    fnr = []    

    for i in xrange(1,101,5):
        trainingSet,testSet = mouthSet.splitInTestAndTraining(frac=.6)
        (a,b,c)=_classify(trainingSet, testSet,n_estimators=i)
        tpr.append(a)
        fpr.append(b)
        fnr.append(c)
    
    plt.figure()
    plt.plot(range(1,101,5),tpr)
    plt.xlabel('number of trees')
    plt.ylabel('score')
    plt.title('.6 training set, max_feature 3, min_split 1' )
   
    fpr=[]
    tpr=[]
    fnr = []
    for i in xrange(1,20):
        trainingSet,testSet = mouthSet.splitInTestAndTraining(frac=i*.05)
        (a,b,c)=_classify(trainingSet, testSet,n_estimators=70)
        tpr.append(a)
        fpr.append(b)
        fnr.append(c)
    
    plt.figure()
    plt.plot(np.arange(0.05,1,0.05),tpr)
    plt.xlabel('fraction used as training set')
    plt.ylabel('score')
    plt.title('ntrees = 70, max_feature 3, min_split 1' )
    
    fpr=[]
    tpr=[]
    fnr = []
    for i in xrange(1,11):
        trainingSet,testSet = mouthSet.splitInTestAndTraining(frac=.7)
        (a,b,c)=_classify(trainingSet, testSet,n_estimators=70,max_features=i)
        tpr.append(a)
        fpr.append(b)
        fnr.append(c)
    
    plt.figure()
    plt.plot(range(1,11),tpr)
    plt.xlabel('number of features used')
    plt.ylabel('score')
    plt.title('ntrees = 70,frac=.7 min_split 1' )
    
    
    plt.show()
    
    #classifier
    #linSVM = svm.SVC(kernel = 'linear',C=1)
    
    #this takes forever: check if that can be true
    #scoresLinSVM = cross_validation.cross_val_score(linSVM,data,y=targetNum,n_jobs=-1,verbose = 1)
    
    #implement random forest classifier with verbosity level
    
    #scoresRF = cross_validation.cross_val_score(rf,mouthSet.data,y=mouthSet.targetNum,n_jobs=nJobs,verbose = 1)
    #print(scoresRF)    
    return

def _classify(trainingSet, testSet,plotting = False,n_estimators=60,max_features=3):
    rf = RandomForestClassifier(n_estimators=n_estimators, max_features =max_features ,max_depth=None,min_split=1, random_state=0,n_jobs=1)    
    
    rf=rf.fit(trainingSet.data,trainingSet.targetNum)
   
    print(rf.score(testSet.data,testSet.targetNum))
    out = rf.predict(testSet.data);
    targetNum = np.array(testSet.targetNum);
    totalNum =float(len(targetNum))
    tp = np.count_nonzero(out==targetNum)
    
    tpr=tp/totalNum
    print(str(tp) + ' ' +str(totalNum))
    print(tpr)
    
    fp = np.count_nonzero((out==1) & (targetNum == 0))
    fpr=fp/totalNum
    fn = totalNum-fp-tp  
    fnr = fn/totalNum
    print('Fraction Correct: {0:.2f}\nfraction fals open: {1:.2f} \n fraction false closed: {2:.2f} '.format(tpr,fpr,fnr))
    if (plotting):    
        testSet.classifiedAs = map(utils.two2MouthLabel,out)
        viewer=plottingUtils.ClassifiedImViewer(path_ea+'/grayScale/',testSet,suffix='_0.png')
        viewer.view(comparer=plottingUtils.MouthTwo2FourComparer,grayScale=True)
    return (tpr,fpr,fnr)

        
if __name__=='__main__':
    njobs = sys.argv[1]
    main(int(njobs))
