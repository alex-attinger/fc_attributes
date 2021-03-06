# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:32:34 2013

@author: attale00

"""
import cv2
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

def getFileNamesFP(data):
    '''
    data: list of tuples with t[0]: label
                                t[1]:probability
                                
    '''
    data = sorted(data,lambda x,y: cmp(y[1],x[1]))    
#    prob = [p[1] for p in data]
#    labels = [p[0] for p in data]
#    fn = [p[2] for p in data]
#    
#    names =[]
#    idx = 0    
#    
#    for i in range(n):
#        out = prob.index(0,idx)

    return [(v,n) for n,v in enumerate(data) if v[0]==0]
    
def getFileNamesFN(data):
    '''
    data: list of tuples with t[0]: label
                                t[1]:probability
                                
    '''
    data = sorted(data,lambda x,y: cmp(x[1],y[1]))

    return [(v,n) for n,v in enumerate(data) if v[0]==1]
    
    

def mirror(src_dir,file_in=[],dest='.',roi=None):
    for f in file_in:
        try:
            im=cv2.imread(src_dir+'/'+f,-1)
            
            if roi is not None:
                ex = im[roi[0]:roi[1],roi[2]:roi[3]]
            else:
                ex = im
                
            flipped = cv2.flip(ex,1)
            cv2.imwrite(dest+'/'+f,flipped)
        except Exception as e:
            print 'did not work for: '+f
            print (e.message)
                

def make8Bit(src_dir,file_in=[], dest = '.',normalized='',resize=None,color=False):
    for f in file_in:
        try:
            im = cv2.imread(src_dir+'/'+f,-1)
            if not color and len(np.shape(im))==3:
                im = cv2.cvtColor(im,cv2.cv.CV_RGB2GRAY)
            name = dest+'/'+f
            if normalized=='hist':
                im=cv2.equalizeHist(im)
            if resize is not None:
                im = cv2.resize(im,resize)
            
            cv2.imwrite(name,im)
        except Exception as e:
            print('did not work for '+f)
            print(e.message)
            

def makeGradientImags(src_dir,file_in=[],destX='./gradX/',destY='./gradY/',destMag='./mag/',destDir='./direction/'):
    for f in file_in:
        try:
            im = cv2.imread(src_dir+'/'+f,-1)
            im = cv2.cvtColor(im,cv2.cv.CV_RGB2GRAY)
            
            #the gradients
            gradX=cv2.Sobel(im,cv2.CV_32F,1,0)
            gradY=cv2.Sobel(im,cv2.CV_32F,0,1)
            #magnitude
            mag=np.hypot(gradY,gradX)
            #direction
            dire=np.arctan2(gradY,gradX)
            #normalizing and writing
            gradX=cv2.normalize(gradX,gradX,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
            cv2.imwrite(destX+f,gradX)
            
            gradY=cv2.normalize(gradY,gradY,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
            cv2.imwrite(destY+f,gradY)
            
            dire=cv2.normalize(dire,dire,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
            cv2.imwrite(destDir+f,dire)
            
            mag=cv2.normalize(mag,mag,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
            cv2.imwrite(destMag+f,mag)
        except Exception as e:
            print('did not work for '+f)
            print(e.message)
            

        
def mapLabelToNumbers(labelList,mapping):
    out = []
    for i in xrange(0,len(labelList)):
        out.append(mapping[labelList[i]])
    return out
  
def mapGlassesLabels(label):
    '''
    maps label name strings to 4 numbers, to be used with map function
    '''
    if label == 'none':
        return 0
    if label == 'light' or label=='thin':
        return 1
    if label == 'thick':
        return 2
    raise Exception(label +' does not have a mapping')
def maptwo2GlassesLabel(label):
    if label == 0:
        return 'no glasses'
    if label == 1:
        return 'glasses'
  
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
    if label in ['closed','close']:
        return 0
    if label == 'narrow':
        return 0 # usually 0
    if label == 'open':
        return 1
    if label in ['wideOpen','wide-open']:
        return 1
    raise Exception(label +' does not have a mapping')

def mapSexLabel2Two(label):
    if label == 'male':
        return 0
    if label == 'female':
        return 1

def two2MouthLabel(label):
    if label == 0:
        return 'closed'
    if label == 1:
        return 'open'

def four2MouthLabel(label):
    if label == 0:
        return 'closed'
    if label == 1:
        return 'narrow'
    if label == 2:
        return 'open'
    if label == 3:
        return 'wideOpen'
    
def dilate(im):
        
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilation_size,dilation_size))
    dilated = cv2.dilate(img,kernel)
    cv2.imshow('dilation demo',dilated)



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
        if(os.path.isfile(folder +'/'+f) and not f.startswith('_')):
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
        try:
            val = parseSingleLabelFile(name,labelname)
            labels[fil]=val
        except Exception as e:
            print e
        
    
    return labels
    
def addDatasetHDFFile(folder,hdfFiles,datasetpath,datatype,dataf):
    for f in hdfFiles:
        name = folder + '/'+f
        hf=h5py.File(name)
        hf.create_dataset(datasetpath,[np.size(dataf)],dtype=datatype,data=dataf[f])
        hf.close()
        