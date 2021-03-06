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


path = '/local/attale00/AFLW_ALL/'
path_ea = '/local/attale00/AFLW_cropped/multiPIE_cropped3/'
path_label = '/local/attale00/a_labels'


allLabelFiles =  utils.getAllFiles(path_label)
    
labeledImages = [i[0:16]+'.png' for i in allLabelFiles]




def multiPiePose():



    path_pose = '/local/attale00/poseLabels/multipie'
    
    poseLabel = utils.parseLabelFiles(path_pose,'rotY',labeledImages,cutoffSeq='.png',suffix='.frog')
    
    poses = [(k,np.double(v)*180/np.pi) for k,v in poseLabel.iteritems()]
    poses = sorted(poses,key=lambda s: s[1])
    return poses
    
def multiPiePoseOriginal():
    p={'140':'-15', '080':'-45', '190':'45', '130':'-30', '050':'15', '051':'0', '041':'30'}
    l=['080','130','140','051','050','041','190']
    poses = [(k,np.int(p[k[10:13]])) for k in labeledImages]
    poses = sorted(poses,key=lambda s: s[1])
    return poses


def splitByPose(poses,binmax=90,stepsize = 20):
    
    p=np.array([i[1] for i in poses])
    
    bins = range(-binmax,binmax,stepsize)
    
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

    
    poses = multiPiePose()
    #poses = multiPiePoseOriginal()
    poseDict=splitByPose(poses,binmax=75, stepsize = 30)

    
    for k,v in poseDict.iteritems():
        fn = [i[0] for i in v]
    
        labs=utils.parseLabelFiles(path_label,'mouth',fn,cutoffSeq='.png',suffix='_face0.labels')
        labs=dict((lk,lv) for (lk,lv) in labs.iteritems() if not lv.startswith('narr'))
    
    
        testSet = fg.dataContainer(labs)
    
    
 
        fg.getHogFeature(testSet,None,path=path_ea,ending='.png',extraMask = None,orientations = 9, cells_per_block=(3,3),pixels_per_cell=(24,8),maskFromAlpha=False)

   
            
    
        testSet.targetNum=map(utils.mapMouthLabels2Two,testSet.target)
        rf=classifierUtils.standardRF(max_features = 30,min_split=12,max_depth=70)
 
        if mode in ['s','v']:
            print 'Classifying with loaded classifier'
            print '------------------- pose {}-----------------'.format(k)
            plt.figure()
            #obj=classifierUtils.classifyWithOld(path,testSet,mode,clfPath = '/home/attale00/Desktop/classifiers/thesis/poseSplit/pose{}'.format(k))
            obj=classifierUtils.classifyWithOld(path,testSet,mode,clfPath = '/home/attale00/Desktop/classifiers/thesis/mirror/rfHogMirror'.format(k))
            obj.plot(title='Mirrored,Pose: {}, ntot: {}, nOpen{}'.format(k,len(testSet.data),testSet.targetNum.count(1)))
            pickle.dump(obj,open('multiPie_mirror_aggregate{}'.format(k),'w'))
        elif mode in ['c']:
            print 'cross validation of data'
            rValues = classifierUtils.dissectedCV(rf,testSet)
            plt.title('Pose: {}, n: {}'.format(k,len(v)))
            #pickle.dump(rValues,open('patches_pose_hog_{}'.format(k),'w'))
        elif mode in ['save']:
            print 'saving new classifier'
            _saveRF(testSet,rf,identifier = k)
   
        
def _saveRF(testSet,rf,filters=None,meanI=None,identifier=''):
   
    rf.fit(testSet.data,testSet.targetNum)
    root='/home/attale00/Desktop/classifiers/thesis/poseSplit/'
    
    pickle.dump(rf,open(root+'pose'.format(identifier),'w'))
    
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

