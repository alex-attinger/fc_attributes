# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:35:54 2013

@author: attale00
"""

## mouth

import utils
from cv2 import imread
import numpy as np

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
    
    #get grayscale file
    mouth_open = []
    mouth_closed = []
    k = labs.keys();
    
    for f in k[1:10]:
        prefix = f.split('.')[0]
        f_name = path_ea+'/grayScale/'+prefix+'_0.png'
        im = imread(f_name)
        roi = im[0:452,60:452]
        vals,bins = np.histogram(roi,bins=10,range=(1,255))
        label = labs[f]
        if(label=='open' or label=='wideOpen'):
            mouth_open.append(vals)
        elif(label=='closed'):
            mouth_closed.append(vals)
    return(mouth_open,mouth_closed)
        
if __name__=='__main__':
    main()