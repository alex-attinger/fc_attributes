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
import matplotlib.pyplot as plt

def main():
    poses_a = aflwPose()
    poses_m = multiPiePose()
    
    binrange = (-75,75)
    nbins = 15
    normed = True
    
    (count_a,edge)=np.histogram(poses_a,bins =nbins,range=binrange,density = normed) 
    (count_m,edge)=np.histogram(poses_m,bins = nbins,range=binrange,density = normed)
    plt.bar(edge[0:-1],count_a,width = 4,color='b',label='AFLW')
    plt.bar(edge[0:-1]+5,count_m,width = 4,color='r',label='Multi-PIE')
    plt.legend()
    plt.xlabel('Angle')
    plt.ylabel('normalized counts')
    plt.xlim((-80,80))
    plt.xticks(range(-70,76,10),range(-70,76,10))    
    
    binrange=(0,70)
    nbins = 7    
    plt.figure()
    (count_a,edge)=np.histogram(np.abs(poses_a),bins =nbins,range=binrange,density = normed) 
    (count_m,edge)=np.histogram(np.abs(poses_m),bins = nbins,range=binrange,density = normed)
    plt.bar(edge[0:-1],count_a,width = 4,color='b',label='AFLW')
    plt.bar(edge[0:-1]+4,count_m,width = 4,color='r',label='Multi-PIE')
    plt.legend()
    plt.xlabel('Angle')
    plt.ylabel('normalized counts')
    plt.xlim((0,80))
    plt.xticks(range(0,76,10),range(0,76,10)) 
    
    
    
    plt.show()
    
    

    return (count_a,count_m,edge)
    
    

def aflwPose():

    path_ea = '/local/attale00/AFLW_cropped/mouth_img_error/'
#    
    fileNames = utils.getAllFiles(path_ea);

    
    
    path_pose = '/local/attale00/poseLabels/aflw'
    
    poseLabelaflw = utils.parseLabelFiles(path_pose,'rotY',fileNames,cutoffSeq='.png',suffix='.frog')
    
    poses_aflw = [np.double(i)*180/np.pi for i in poseLabelaflw.values()]
    
    

    return np.array(poses_aflw)

def multiPiePose():

    

   
    allLabelFiles =  utils.getAllFiles('/local/attale00/a_labels')
    
    labeledImages = [i[0:16]+'.png' for i in allLabelFiles]
    
        
   
    path_pose = '/local/attale00/poseLabels/multipie'
    
    poseLabel = utils.parseLabelFiles(path_pose,'rotY',labeledImages,cutoffSeq='.png',suffix='.frog')
    
    poses = np.array([np.double(i)*180/np.pi for i in poseLabel.values()])
    return poses
    
    
if __name__=='__main__':
    main()