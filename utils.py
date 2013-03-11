# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:32:34 2013

@author: attale00

"""
import cv2
import os
import numpy as np

def make8BitGray(src_dir,file_in=[], dest = '.'):
    for f in file_in:
        im = cv2.imread(src_dir+'/'+f,0)
        name = dest+'/'+f
        cv2.imwrite(name,im)


def emptyMouthPixels(src_folder,file_list):
    out = cv2.imread(src_folder+'/'+file_list[0])
    out = np.zeros(out.shape)
    for f in file_list:
        out = out+cv2.imread(src_folder+'/'+f)
    return out

def getAllFiles(folder):
    files = os.listdir(folder)
    out = []
    
    for f in files:
        if(os.path.isfile(folder +'/'+f)):
            out.append(f)
    
    return out
    