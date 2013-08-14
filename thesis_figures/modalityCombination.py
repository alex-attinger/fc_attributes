# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:20:26 2013

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

allLabelFiles =  utils.getAllFiles('/local/attale00/a_labels')

labeledImages = [i[0:16]+'.png' for i in allLabelFiles]

#labs=utils.parseLabelFiles(path+'/Multi-PIE/labels','mouth',labeledImages,cutoffSeq='.png',suffix='_face0.labels')
labs=utils.parseLabelFiles('/local/attale00/a_labels','mouth',labeledImages,cutoffSeq='.png',suffix='_face0.labels')
labs=dict((k,v) for (k,v) in labs.iteritems() if not v.startswith('narr'))
    

def patches():
    path = '/local/attale00/AFLW_ALL/'
    path_ea = '/local/attale00/AFLW_cropped/multiPIE_cropped3/'
    

   
   
   
    
    
    testSet = fg.dataContainer(labs)
    
    
    roi=(0,37,0,115)
    roi=None    



            
 
    X=fg.getAllImagesFlat(path_ea,testSet.fileNames,(40,120),roi=roi)

    W=np.load('/home/attale00/Desktop/classifiers/thesis/filtercombined.npy')
    m=np.load('/home/attale00/Desktop/classifiers/thesis/meancombined.npy')
    X1=X-m
    data=np.dot(X1,W.T)    
    
    for i in range(len(testSet.fileNames)):
            testSet.data[i].extend(data[i,:])

    fg.getHogFeature(testSet,roi,path=path_ea,ending='.png',extraMask = None,orientations = 9, pixels_per_cell=(24,8),cells_per_block=(3,3),maskFromAlpha=False)
    fg.getColorHistogram(testSet,roi,path=path_ea,ending='.png',colorspace='lab',bins=40)

    
   
            
    
    testSet.targetNum=map(utils.mapMouthLabels2Two,testSet.target)
    

    print 'Classifying with loaded classifier'
    mode = 's'
    classifierUtils.classifyWithOld(path,testSet,mode,clfPath = '/home/attale00/Desktop/classifiers/thesis/combined')       
    return testSet
   
def texture():
    path = '/local/attale00/extracted_pascal__4__Multi-PIE'
    path_ea = path+'/color128/'
   
    
    testSet = fg.dataContainer(labs)    
    roi=(50,74,96,160)
    X=fg.getAllImagesFlat(path_ea,testSet.fileNames,(128,256),roi=roi)


        
    
    W=np.load('/home/attale00/Desktop/classifiers/thesis/texture/filtercombined.npy')
    m=np.load('/home/attale00/Desktop/classifiers/thesis/texture/meancombined.npy')
    X1=X-m
    data=np.dot(X1,W.T)    

    for i in range(len(testSet.data)):
        testSet.data[i].extend(data[i,:])
##    
    
   


  
    fg.getHogFeature(testSet,roi,path=path_ea,ending='.png',extraMask = None,orientations = 4, cells_per_block=(6,2),maskFromAlpha=False)
    fg.getColorHistogram(testSet,roi,path=path_ea,ending='.png',colorspace='lab',bins=40)    
    testSet.targetNum=map(utils.mapMouthLabels2Two,testSet.target)
    
   
    mode = 's'
    print 'Classifying with loaded classifier'
    classifierUtils.classifyWithOld(path,testSet,mode,clfPath = '/home/attale00/Desktop/classifiers/thesis/texture/combined')
    return testSet

def errorpatch():
    path = '/local/attale00/AFLW_ALL/'
    path_ea = '/local/attale00/AFLW_cropped/mouth_img_error_multiPie/'
    


    
    
    testSet = fg.dataContainer(labs)
    
  
    fg.getImagePatchStat(testSet,path=path_ea,patchSize=(4,12),overlap = 2,mode=0)
  
 
   
            
    
    testSet.targetNum=map(utils.mapMouthLabels2Two,testSet.target)
  
    print 'Classifying with loaded classifier'
    mode = 's'
    classifierUtils.classifyWithOld(path,testSet,mode,clfPath = '/home/attale00/Desktop/classifiers/thesis/errorpatches/mode_0rferror')
    return testSet
    
    
if __name__=='__main__':
    testSets = {'patches':None,'errorpatch':None,'texture':None}
    rocObj = {}
    
    for k in testSets:
        testSets[k]=locals()[k]()
        
    for k,v in testSets.iteritems():
        plt.figure()
        rocObj[k]=classifierUtils.ROCCurve(v)
        rocObj[k].plot()

    labels = testSets['patches'].targetNum
    probabilities = np.zeros((len(testSets['patches'].targetNum,)))
#        
#    for i,v in enumerate(labels):    
#        probabilities[i]=sum([.3*v.probabilities[i][1] for v in testSets.itervalues() ] )
#    rocObject=PyRoc.ROCData(zip(labels,probabilities,testSets['patches'].fileNames))
#    plt.figure()
#    rocObject.plot()
#    pickle.dump(rocObject,open('modality_combo','w'))
    
        
    for i,v in enumerate(labels):    
        probabilities[i]=sum([.5*v.probabilities[i][1] for v in [testSets['patches'],testSets['errorpatch']] ] )
    rocObject=PyRoc.ROCData(zip(labels,probabilities,testSets['patches'].fileNames))
    plt.figure()
    rocObject.plot()
    pickle.dump(rocObject,open('modality_errorpatch_patch','w'))
    
    for i,v in enumerate(labels):    
        probabilities[i]=sum([.5*v.probabilities[i][1] for v in [testSets['patches'],testSets['texture']] ] )
    rocObject=PyRoc.ROCData(zip(labels,probabilities,testSets['patches'].fileNames))
    plt.figure()
    rocObject.plot()
    pickle.dump(rocObject,open('modality_texture_patch','w'))
    
    for i,v in enumerate(labels):    
        probabilities[i]=sum([.5*v.probabilities[i][1] for v in [testSets['errorpatch'],testSets['texture']] ] )
    rocObject=PyRoc.ROCData(zip(labels,probabilities,testSets['patches'].fileNames))
    plt.figure()
    rocObject.plot()
    pickle.dump(rocObject,open('modality_errorpatch_texture','w'))
    
    
    
    #plt.savefig('/home/attale00/Dropbox/tex_thesis/images/modality_combo.pdf')
#    fp=utils.getFileNamesFP(rocObject.data)
#    fn = utils.getFileNamesFN(rocObject.data)
#    print 'false positives:'
#    for p in fp[:5]:
#        print p
#    
#    print 'false negatives:'
#    for p in fn[:5]:
#        print p
#    
#    plottingUtils.coloredProbabilityDistribution(rocObject.data)
#    
    plt.show()
    