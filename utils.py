# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:32:34 2013

@author: attale00

"""
import cv2
import os
import h5py
import numpy as np

def make8BitGray(src_dir,file_in=[], dest = '.'):
    for f in file_in:
        im = cv2.imread(src_dir+'/'+f,0)
        name = dest+'/'+f
        cv2.imwrite(name,im)
        
def mapLabelToNumbers(labelList,mapping):
    out = []
    for i in xrange(0,len(labelList)):
        out.append(mapping[labelList[i]])
    return out
    
def mapMouthLabels2Four(label):
    '''
    maps label name strings to 4 numbers, to be used with map function
    '''
    if label == 'closed':
        return 0
    if label == 'narrow':
        return 1
    if label == 'open':
        return 2
    if label == 'wideOpen':
        return 3
    raise Exception(label +' does not have a mapping')
    
def mapMouthLabels2Two(label):
    '''
    maps label name strings to 2 numbers, to be used with map function
    '''
    if label == 'closed':
        return 0
    if label == 'narrow':
        return 0
    if label == 'open':
        return 1
    if label == 'wideOpen':
        return 1
    raise Exception(label +' does not have a mapping')




def emptyMouthPixels(src_folder,file_list):
    out = cv2.imread(src_folder+'/'+file_list[0],0)
    out = np.zeros(out.shape)
    for f in file_list:
        tmp = cv2.imread(src_folder+'/'+f,0)
        if not (tmp is None):
            try:
                out = out+tmp
            except ValueError:
                print(f)
        else:
            print(f)
    return out

def getAllFiles(folder):
    files = os.listdir(folder)
    out = []
    
    for f in files:
        if(os.path.isfile(folder +'/'+f)):
            out.append(f)
    
    return out
    
def _makehdfFile(path):
    f=h5py.File(path)
    f.close

def makehdfFiles(folder,filenames):
    for f in filenames:
        name = f.split('.')[0]
        name = folder + '/'+name+'.hdf5'
        _makehdfFile(name)

def parseLabelINIFile(filepath,labelname):
    lFile = file(filepath);
    linetag = labelname+'.values'
    for line in lFile:
        if line.startswith(linetag):
            vals = line.split('=')[1]
            vals = [x.strip() for x in vals.split(',')]
    lFile.close()
    return vals
    
def parseSingleLabelFile(fileName,label):
    f=file(fileName)
    for line in f:
        if line.startswith(label+'='):
            val = line.split('=')[1]
            val = val.strip()
            f.close()
            return val
    raise Exception('File '+fileName+' did not contain attribute: '+label)
        
        
def parseLabelFiles(folder,labelname,filenames,cutoffSeq='0.png',suffix='face0.labels'):
    """
    Parse the label Files
    :folder      -- folder containing the label files
    :labelname   -- name of label, e.g. mouth
    :filenames   -- list of all the files
    :cutoffSeq   -- specify this if names of labelfiles are not just filename + suffix
    :suffix      -- suffix for the label files 
    """
    labels = dict()
    cutofflen = len(cutoffSeq)
    for fil in filenames:
        name = folder +'/'+fil[0:len(fil)-cutofflen]+suffix
        val = parseSingleLabelFile(name,labelname)
        labels[fil]=val
    
    return labels
    
def addDatasetHDFFile(folder,hdfFiles,datasetpath,datatype,dataf):
    for f in hdfFiles:
        name = folder + '/'+f
        hf=h5py.File(name)
        hf.create_dataset(datasetpath,[np.size(dataf)],dtype=datatype,data=dataf[f])
        hf.close()
        