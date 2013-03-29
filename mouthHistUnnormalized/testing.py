# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:38:38 2013

@author: attale00
"""

import cv2
import numpy as np
import utils
path = '/local/attale00/GoodPose'
pathg = path+'/extracted_alpha/grayScale'

files = utils.getAllFiles(pathg)
cv2.namedWindow('thresh')
big = np.zeros((512,2*512),dtype='uint8')
for f in files:
    im = cv2.imread(pathg+'/'+f,-1)
    big[:,0:512]=im
    ret,imbw = cv2.threshold(im,155,255,cv2.THRESH_BINARY)
    big[:,512:1024]=imbw
    cv2.imshow('test',big)
    cv2.waitKey(0)

print('done')