# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:39:45 2013

@author: attale00
"""

import utils
import featureGeneration as fg
import cv2
import numpy as np
import pickle
import sys
import plottingUtils
import classifierUtils

def main():
    path='/local/attale00/'
    
    allFiles = utils.getAllFiles(path+'Multi-PIE/extracted')
    
    allLabelFiles = utils.getAllFiles(path+'Multi-PIE/db_labels')
    np.random.shuffle(allLabelFiles)
    #get the labels from the database, for each person
    sliceat=250
    labelstest=utils.parseLabelFiles(path+'Multi-PIE/db_labels','sex',allLabelFiles[0:sliceat],cutoffSeq='',suffix='')
    
    labelstraining = utils.parseLabelFiles(path+'Multi-PIE/db_labels','sex',allLabelFiles[sliceat:],cutoffSeq='',suffix='')
    #now generate the label dict for each file
    
    labsTest={}
    labsTraining={}
    for f in allFiles:
        if labelstest.has_key(f[0:3]+'.labels'):
            labsTest[f]=labelstest[f[0:3]+'.labels']
        elif labelstraining.has_key(f[0:3]+'.labels'):
            labsTraining[f]=labelstraining[f[0:3]+'.labels' ]
            
    testSet = fg.dataContainer(labsTest)
    trainingSet = fg.dataContainer(labsTraining)
    
    roi=(0,64,0,64)
    ppc=(8,8)
    cpb=(8,8)
    
    fg.getHogFeature(testSet,roi,path=path+'Multi-PIE_grayScale64/',ending=None,extraMask = None,pixels_per_cell=ppc,cells_per_block=cpb)
    
    fg.getHogFeature(trainingSet,roi,path=path+'Multi-PIE_grayScale64/',ending=None,extraMask = None,pixels_per_cell=ppc, cells_per_block=cpb)
    
    testSet.targetNum=map(utils.mapSexLabel2Two,testSet.target)
    trainingSet.targetNum = map(utils.mapSexLabel2Two,trainingSet.target)
    
    rf1=classifierUtils.standardRF(max_features=np.sqrt(len(testSet.data[0])))
    rf2=classifierUtils.standardRF(max_features=np.sqrt(len(trainingSet.data[0])))
    
    rf1.fit(testSet.data,testSet.targetNum)
    
    s=rf1.score(trainingSet.data,trainingSet.targetNum)
    trainingSet.classifiedAs=rf1.predict(trainingSet.data)
    trainingSet.hasBeenClassified=True
    classifierUtils.evaluateClassification(trainingSet,{0:'male',1:'female'})
    
    
    
    print 'Score: {}'.format(s)
    
    print '----------other way around ----\n'
    
    rf2.fit(trainingSet.data,trainingSet.targetNum)
    
    s=rf2.score(testSet.data,testSet.targetNum)
    testSet.classifiedAs=rf2.predict(testSet.data)
    testSet.hasBeenClassified=True
    classifierUtils.evaluateClassification(testSet,{0:'male',1:'female'})
    print 'Score: {}'.format(s)
    

if __name__=='__main__':
    main()