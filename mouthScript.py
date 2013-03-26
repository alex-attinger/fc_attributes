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

def main():

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
    data,target = fg.getHistogram(10,(0,xx,xx,xx),labelFileDict = labs,path = pathea+'/grayScale/',ending='_0.png')

    #classifier
    classifier = svm.SVC(kernel = 'linear')
    [classifier.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test])
...          for train, test in kfold]
    return
        
if __name__=='__main__':
    main()
