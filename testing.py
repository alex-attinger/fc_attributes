# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:38:38 2013

@author: attale00
"""

import cv2
import numpy as np
import utils


def plotThreshold(path='/local/attale00/GoodPose/extracted_alpha/grayScale',thresh=100):


    files = utils.getAllFiles(path)
   
    big = np.zeros((512,2*512),dtype='uint8')
    for f in files:
        im = cv2.imread(path+'/'+f,-1)
        big[:,0:512]=im
        ret,imbw = cv2.threshold(im,thresh,255,cv2.THRESH_BINARY)
        big[:,512:1024]=imbw
        cv2.imshow('test',big)
        k=cv2.waitKey(0)
        if k==27:
            
            return

    print('done')
    
    

def plotExcerpt(path='/local/attale00/GoodPose/extracted_alpha/grayScale',roi=(0,250,60,452)):


    files = utils.getAllFiles(path)
    cv2.namedWindow('thresh')
    big = np.zeros((512,2*512),dtype='uint8')
    for f in files:
        im = cv2.imread(path+'/'+f,-1)
        im = cv2.cvtColor(im,cv2.cv.CV_RGB2GRAY)
        big[:,0:512]=im
        
        big[roi[0]:roi[1],512:512+roi[3]-roi[2]]=im[roi[0]:roi[1],roi[2]:roi[3]]
        cv2.imshow('test',big)
        k=cv2.waitKey(0)
        if k==27:
            return

    print('done')
    