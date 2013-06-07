# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:49:39 2013

@author: attale00
"""

import utils
import os

labelPath='/export/somalia/egger/aflw_all/quality/labels'
#sourcePath='/export/somalia/egger/aflw_all/montages'
sourcePath='/export/somalia/egger/aflw_all'
targetPath ='/local/attale00/AFLW_ALL/montage'
targetPath='/local/attale00/AFLW_ALL/original'

labelFiles=utils.getAllFiles(labelPath)
counter =0
for labelFile in labelFiles:
    f=open(labelPath+'/'+labelFile)
    lines = f.readlines()
    
    if lines[1].startswith('fit=good') or lines[4].startswith('fit_mouth=good'):
        counter +=1
        os.symlink(sourcePath+'/'+labelFile[:-13]+'.png',targetPath+'/'+labelFile[:-13]+'.png')
    
print counter
        