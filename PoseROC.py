# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 10:25:06 2013

@author: attale00
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:29:01 2013

@author: attale00
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:24:39 2013

This script classifies the multipie pictures with the random forest classifer learned on the aflw database pics
. It was trained on hog features only on the down scaled images. see the accompanying info file to the classfier for details

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
from sklearn import svm
from sklearn import linear_model
from sklearn.decomposition import FastICA  
from scipy import linalg
import PyRoc
import matplotlib.pyplot as plt

def patches():
    path = '/local/attale00/AFLW_ALL/'
    path_ea = '/local/attale00/AFLW_cropped/multiPIE_cropped3/'
    

   
    allLabelFiles =  utils.getAllFiles('/local/attale00/a_labels')
    
    labeledImages = [i[0:16]+'.png' for i in allLabelFiles]
   
    labs=utils.parseLabelFiles('/local/attale00/a_labels','mouth',labeledImages,cutoffSeq='.png',suffix='_face0.labels')

    fileNames = labeledImages;
    testSet = fg.dataContainer(labs)
    
    
    roi=(0,37,0,115)
    roi=None    
 


            
 
    X=fg.getAllImagesFlat(path_ea,testSet.fileNames,(40,120),roi=roi)
 
   
    
    W=np.load('/home/attale00/Desktop/classifiers/patches/filter2.npy')
    m=np.load('/home/attale00/Desktop/classifiers/patches/meanI2.npy')
    X1=X-m
    data=np.dot(X1,W.T)    
    
    for i in range(len(fileNames)):
            testSet.data[i].extend(data[i,:])

    
    fg.getHogFeature(testSet,roi,path=path_ea,ending='.png',extraMask = None,orientations = 5, pixels_per_cell=(24,8),cells_per_block=(3,3),maskFromAlpha=False)
    fg.getColorHistogram(testSet,roi,path=path_ea,ending='.png',colorspace='lab',bins=20)
   
   
            
    
    testSet.targetNum=map(utils.mapMouthLabels2Two,testSet.target)
    clfPath = '/home/attale00/Desktop/classifiers/patches/rfICAHogColor'
    f=file(clfPath,'r')
    print 'classifier used: '+ f.name
    clf = pickle.load(f)
    testSet.classifiedAs=clf.predict(testSet.data)
    testSet.probabilities=clf.predict_proba(testSet.data)      
        

    
    return testSet

def texture():
    path = '/local/attale00/extracted_pascal__4__Multi-PIE'
    path_ea = path+'/color128/'
   
    allLabelFiles =  utils.getAllFiles('/local/attale00/a_labels')
    
    labeledImages = [i[0:16]+'.png' for i in allLabelFiles]
    
    
    labs=utils.parseLabelFiles('/local/attale00/a_labels','mouth',labeledImages,cutoffSeq='.png',suffix='_face0.labels')
    
        
    testSet = fg.dataContainer(labs)    
    roi=(50,74,96,160)
    X=fg.getAllImagesFlat(path_ea,testSet.fileNames,(128,256),roi=roi)


    W=np.load('/home/attale00/Desktop/classifiers/ica/filter1.npy')
    m=np.load('/home/attale00/Desktop/classifiers/ica/meanI1.npy')
    X1=X-m
    data=np.dot(X1,W.T)    
    
    for i in range(len(testSet.data)):
        testSet.data[i].extend(data[i,:])
    
 
  
    fg.getHogFeature(testSet,roi,path=path_ea,ending='.png',extraMask = None,orientations = 3, cells_per_block=(6,2),maskFromAlpha=False)
    fg.getColorHistogram(testSet,roi,path=path_ea,ending='.png',colorspace='lab',bins=10)    
    testSet.targetNum=map(utils.mapMouthLabels2Two,testSet.target)
    
       
    clfPath = '/home/attale00/Desktop/classifiers/ica/rf128ICAHOGCOLOR'
    f=file(clfPath,'r')
    print 'classifier used: '+ f.name
    clf = pickle.load(f)
    testSet.classifiedAs=clf.predict(testSet.data)
    testSet.probabilities=clf.predict_proba(testSet.data)      
        

    
    return testSet
    
def textureICAonly():
    path = '/local/attale00/extracted_pascal__4__Multi-PIE'
    path_ea = path+'/color128/'
   
    allLabelFiles =  utils.getAllFiles('/local/attale00/a_labels')
    
    labeledImages = [i[0:16]+'.png' for i in allLabelFiles]
    
    
    labs=utils.parseLabelFiles('/local/attale00/a_labels','mouth',labeledImages,cutoffSeq='.png',suffix='_face0.labels')
    
        
    testSet = fg.dataContainer(labs)    
    roi=(50,74,96,160)
    X=fg.getAllImagesFlat(path_ea,testSet.fileNames,(128,256),roi=roi)


    W=np.load('/home/attale00/Desktop/classifiers/ica/filter1.npy')
    m=np.load('/home/attale00/Desktop/classifiers/ica/meanI1.npy')
    X1=X-m
    data=np.dot(X1,W.T)    
    
    for i in range(len(testSet.data)):
        testSet.data[i].extend(data[i,:])
    
 
  
    #fg.getHogFeature(testSet,roi,path=path_ea,ending='.png',extraMask = None,orientations = 3, cells_per_block=(6,2),maskFromAlpha=False)
    #fg.getColorHistogram(testSet,roi,path=path_ea,ending='.png',colorspace='lab',bins=10)    
    testSet.targetNum=map(utils.mapMouthLabels2Two,testSet.target)
    
       
    clfPath = '/home/attale00/Desktop/classifiers/ica/rf128ICA_1'
    f=file(clfPath,'r')
    print 'classifier used: '+ f.name
    clf = pickle.load(f)
    testSet.classifiedAs=clf.predict(testSet.data)
    testSet.probabilities=clf.predict_proba(testSet.data)      
    return testSet

    
def errorPatch():
    path = '/local/attale00/AFLW_ALL/'
    path_ea = '/local/attale00/AFLW_cropped/mouth_img_error_multiPie/'
    
    allLabelFiles =  utils.getAllFiles('/local/attale00/a_labels')
    
    labeledImages = [i[0:16]+'.png' for i in allLabelFiles]
    
    labs=utils.parseLabelFiles('/local/attale00/a_labels','mouth',labeledImages,cutoffSeq='.png',suffix='_face0.labels')
        
    
#    
    fileNames = labeledImages;

   
    
    
    testSet = fg.dataContainer(labs)
    
    
 
    fg.getImagePatchStat(testSet,path=path_ea,patchSize=(4,12),overlap = 2)
  
 
    
    testSet.targetNum=map(utils.mapMouthLabels2Two,testSet.target)

    clfPath = '/home/attale00/Desktop/classifiers/errorpatches/rferror'
    f=file(clfPath,'r')
    print 'classifier used: '+ f.name
    clf = pickle.load(f)
    testSet.classifiedAs=clf.predict(testSet.data)
    testSet.probabilities=clf.predict_proba(testSet.data)      
    return testSet
   
   
def splitByPose(testSet):
    poses = classifierUtils.splitByPoseFull(testSet)
    textures = {}
    for k in poses:
        textures[k]=classifierUtils.PyRoc.ROCData(poses[k])
    
    return textures
    
if __name__=='__main__':
    #patchSet=patches()
    textureICAOnlySet = textureICAonly()    
    textureSet=texture()
    #errorPatchSet = errorPatch()
    #patchROC = classifierUtils.ROCCurve(patchSet)
    texRoc = classifierUtils.ROCCurve(textureSet)
    texICARoc = classifierUtils.ROCCurve(textureICAOnlySet)    
    #errorPatchRoc = classifierUtils.ROCCurve(errorPatchSet)
    #PyRoc.plot_multiple_roc([patchROC, texRoc, errorPatchRoc],title='All Poses',labels=['Patch','Texture','ErrorPatch'])
    PyRoc.plot_multiple_roc([texICARoc,texRoc],title='',labels=['Gabor Patches Only','all Features'])    
#    t=splitByPose(textureSet)
#    p=splitByPose(patchSet)
#    e=splitByPose(errorPatchSet)
#    for k in p:
#        PyRoc.plot_multiple_roc([p[k],t[k],e[k]],title=k,labels=['Patch','Texture','ErrorPatch'])
    plt.show()