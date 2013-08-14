# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 10:25:06 2013

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
import matplotlib.pyplot as plt


path_ea = '/local/attale00/AFLW_cropped/cropped3/'
fileNames = utils.getAllFiles(path_ea);

def aflwPose():


    
    path_pose = '/local/attale00/poseLabels/aflw'
    
    poseLabelaflw = utils.parseLabelFiles(path_pose,'rotY',fileNames,cutoffSeq='.png',suffix='.frog')
    
    poses_aflw = [(k,np.double(i)*180/np.pi) for k,i in poseLabelaflw.iteritems()]
    
    poses_aflw = sorted(poses_aflw,key=lambda s: s[1])
    
    return poses_aflw
    


def splitByPose(poses,bins = None):
    
    p=np.array([i[1] for i in poses])
    
    
    
    binNumber = np.digitize(p,bins)
    
    
    poseDict={}
    for i in bins:
        poseDict[i]=[]
    
    for i,v in enumerate(poses):
        bi = binNumber[i]
        bi = bi-1 if i>0 else bi
        poseDict[bins[bi]].append(v)
        
    return poseDict
       
        
        
    
    

def main(mode):
    path = '/local/attale00/AFLW_ALL/'
    path_mirrored='/local/attale00/AFLW_cropped/mirrored/'
    
    poses = aflwPose()
    binmax=100
    stepsize = 40
    bins = range(-binmax,binmax,stepsize)
    poseDict=splitByPose(poses,bins=bins)
    
    testSets = {} 
    #original Part
    for k,v in poseDict.iteritems():
        fn = [i[0] for i in v]
    
        labs=utils.parseLabelFiles(path+'/labels/labels','mouth_opening',fn,cutoffSeq='.png',suffix='_face0.labels')
        
        testSets[k] = fg.dataContainer(labs)
        fg.getHogFeature(testSets[k],None,path=path_ea,ending='.png',extraMask = None,orientations = 9, cells_per_block=(3,3),pixels_per_cell=(24,8),maskFromAlpha=False)
    
    
    #mirrored part
    testSetsM = {}
    nBins = len(bins)
    for k,v in poseDict.iteritems():
        binNumber=bins.index(k)
        oppositeBin = bins[nBins-1-binNumber]
        
        fn=[i[0]+'M' for i in poseDict[oppositeBin]]
        labs = utils.parseLabelFiles(path+'/labels/labels','mouth_opening',fn,cutoffSeq='.pngM',suffix='_face0.labels')
        testSetsM[k] = fg.dataContainer(labs)
        fg.getHogFeature(testSetsM[k],None,path=path_mirrored, ending='.png',orientations = 9, cells_per_block=(3,3),pixels_per_cell=(24,8))    

        
    
    for k,v in poseDict.iteritems():
        testSet = testSets[k]
        testSet.addContainer(testSetsM[k])
            
    
        testSet.targetNum=map(utils.mapMouthLabels2Two,testSet.target)
        rf=classifierUtils.standardRF(max_features = 40,min_split=12,max_depth=70)
 
        if mode in ['s','v']:
            print 'Classifying with loaded classifier'
            _classifyWithOld(path,testSet,mode)
        elif mode in ['c']:
            print 'cross validation of data'
            rValues = classifierUtils.dissectedCV(rf,testSet)
            plt.title('Pose: {}, n: {}'.format(k,len(testSet.data)))
            #pickle.dump(rValues,open('patches_pose_hog_{}'.format(k),'w'))
        elif mode in ['save']:
            print 'saving new classifier'
            _saveRF(testSet,rf,identifier = k)
   
        
def _saveRF(testSet,rf,filters=None,meanI=None,identifier=''):
   
    rf.fit(testSet.data,testSet.targetNum)
    root='/home/attale00/Desktop/classifiers/thesis/poseSplit/'
    
    pickle.dump(rf,open(root+'pose_mirror{}'.format(identifier),'w'))
    
    f=open(root+'rficahogcolor.txt','w')
    f.write('Source Images: AFLWALL')
    f.write('attribute: Mouth')
    f.write('Features: ICA HOg color Errorpatches')
    f.write('100 comps \n')
    f.write('20 color bins \n')
    f.write('ppc 24,8, cpb 3,3 dir 5 \n')
    f.write('ROI:(50,74,96,160)\n')
 
    f.write('labels: none: 0, light,thick: 1\n')
    f.close()
    if filters is not None:
        np.save(root+'filter2',filters)
        np.save(root+'meanI2',meanI)
        


def _classifyWithOld(path,testSet,mode):
    #f=file('/home/attale00/Desktop/classifiers/RandomForestMouthclassifier_1','r')
    f=file('/home/attale00/Desktop/classifiers/patches/rfICAMultiPie','r')
    clf = pickle.load(f)
    testSet.classifiedAs=clf.predict(testSet.data)
    testSet.hasBeenClassified = True
    if mode =='s':
        _score(clf,testSet)
    else:
        _view(clf,testSet,path+'Multi-PIE/extracted/')
        _score(clf,testSet)
     

def _score(clf,testSet):
    score = clf.score(testSet.data,testSet.targetNum)
    testSet.hasBeenClassified = True
    classifierUtils.evaluateClassification(testSet,{0:'closed or narrow',1:'open or wide open'})
    print 'Overall Score: {:.3f}'.format(score)
    return
def _view(clf,testSet,path):
    viewer = plottingUtils.ClassifiedImViewer(path,testSet)
    viewer.view(comparer=plottingUtils.MouthTwo2FourComparer)
    
    
if __name__=='__main__':
    if len(sys.argv)==2:
        if sys.argv[1] in ['s','Score']:
            m = 's'
        elif sys.argv[1] in ['v','View']:
            m='v'
        elif sys.argv[1] in ['c']:
            m='c'
        elif sys.argv[1] in ['save']:
            m='save'
        else:
            print 'Option not supported, valid options are s,v. Now just scoring'
            m='s'
    else:
        m='s'
            
    main(m)
    plt.show()
