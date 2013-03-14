# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:35:54 2013

@author: attale00
"""

## mouth

import utils

path = '/local/attale00/GoodPose'
path_ea = path+'/extracted_alpha'
path_adat = path_ea + '/a_data'

filenames = utils.getAllFiles(path_ea);

attribute = 'mouth'

attribute_values = utils.parseLabelINIFile(path+'/mouth_labels/labels.ini',attribute);

print('------------Attribute: \t'+attribute+' ---------------')
for i in attribute_values:
    print('Value: \t'+i)
    
print('----------------------------')

utils.parseLabelFiles(path+'/mouth_labels','mouth',filenames[1:10],cutoffSeq='.png',suffix='_face0.labels')