# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:35:54 2013

@author: attale00
"""

## mouth

import utils
from cv2 import imread
import numpy as np
import featureGeneration as fg
from sklearn import svm, cross_validation
from sklearn.ensemble import RandomForestClassifier

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
    
    labs=utils.parseLabelFiles(path+'/mouth_labels','mouth',filenames,cutoffSeq='.png',suffix='_face0.labels')
    
    #make 10 bin hist for each mouth
    data,target = fg.getHistogram(10,(0,452,60,452),labelFileDict = labs,path = path_ea+'/grayScale/',ending='_0.png')
    
    targetNum = map(utils.mapMouthLabels2Two,target)


    
    #classifier
    linSVM = svm.SVC(kernel = 'linear',C=1)
    
    #this takes forever: check if that can be true
    #scoresLinSVM = cross_validation.cross_val_score(linSVM,data,y=targetNum,n_jobs=-1,verbose = 1)
    
    #implement random forest classifier with verbosity level
    rf = RandomForestClassifier(n_estimators=60, max_features =8 ,max_depth=None,min_split=1, random_state=0,n_jobs=1)    
    scoresLinSVM = cross_validation.cross_val_score(rf,data,y=targetNum,n_jobs=-1,verbose = 1)
    return scoresLinSVM
        
if __name__=='__main__':
    print(main())
