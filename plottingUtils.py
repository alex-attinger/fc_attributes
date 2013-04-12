# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:06:12 2013

@author: attale00
"""
import matplotlib.pyplot as plt
import numpy as np
import utils
import cv2



    
def viewClassifiedImages(path,dataset,suffix = ".png"):

    t=plt.text(20,-20,'Hello WOrld')
    for i in xrange(0,len(dataset.fileNames)):
        #load image
        name = dataset.fileNames[i].split('.')[0]
        name = name+suffix
        im = plt.imread(path+name)
        plt.imshow(im)
        if dataset.hasBeenClassified:
            text = 'Name: '+name + ', label: '+dataset.target[i]+ ', classified as: '+dataset.classifiedAs[i]
        t.set_text(text)
        plt.draw()
        plt.waitforbuttonpress()  
    
def MouthTwo2FourComparer(target,test):
    if target==test:
        return True
    if target in ['closed','narrow',0]:
        if test in ['closed',0]:
            return True
        else:
            return False
    if target in ['open','wideOpen']:
        if test in['open',1]:
            return True
        else:
            return False
        
        
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
    im = cv2.imread(path+'/'+files[0],-1)    
    sh=np.shape(im)
    cv2.namedWindow('Panorama')
    big = np.zeros((sh[0],sh[1]*2),dtype='uint8')
    for f in files:
        im = cv2.imread(path+'/'+f,-1)
        try:
            im = cv2.cvtColor(im,cv2.cv.CV_RGB2GRAY)
        except Exception as e:
            pass
        big[:,0:sh[1]]=im
        
        big[roi[0]:roi[1],sh[1]:sh[1]+roi[3]-roi[2]]=im[roi[0]:roi[1],roi[2]:roi[3]]
        cv2.imshow('test',big)
        k=cv2.waitKey(0)
        if k==27:
            return

    print('done')
    

class ClassifiedImViewer:
    def __init__(self,path,dataset,suffix = '.png'):
        self.path = path
        self.dataset=dataset
        self.suffix = suffix
        self.counter = 0;
        self.comparer = None;
        
    def _event_fct(self,event):
        """ evaluate key-press events """
        
        if event is None:
            self.counter = 0
        else:
            if event.key in ['q', 's']:
                return
            elif self.counter==len(self.dataset.data)-1: 
                return
            elif event.key in ['p']:
                self.counter -= 1
            else:
                self.counter += 1
           
        name = self.dataset.fileNames[self.counter].split('.')[0]
        name = name+self.suffix
        im = plt.imread(self.path+name)
        self.ax.cla()
        self.ax.imshow(im)
        if self.dataset.hasBeenClassified:
            text = 'Name: '+name + ', label: '+self.dataset.target[self.counter]+ ', classified as: '+self.dataset.classifiedAs[self.counter]
        else:
            text = 'Name: '+name + ', label: '+self.dataset.target[self.counter]

        self.ax.set_title(text)
        if(self.comparer!= None):
            if(self.comparer(self.dataset.target[self.counter],self.dataset.classifiedAs[self.counter])):
                self.fig.set_facecolor('g')
            else:
                self.fig.set_facecolor('r')
        if self.grayScale:
            plt.gray()
        plt.draw()
        
        
    def view(self,comparer=None,grayScale=False):
        self.grayScale= grayScale
        self.comparer = comparer
        self.fig = plt.figure("Image Viewer")
        self.ax = plt.subplot(111)
        self._event_fct(None) # initial setup
        plt.connect('key_press_event', self._event_fct)
        plt.show()
        
    