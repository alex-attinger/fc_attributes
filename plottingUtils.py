# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:06:12 2013

@author: attale00
"""
import matplotlib.pyplot as plt
import numpy as np



    
def viewClassifiedImages(path,dataset,suffix = ".png"):

    t=plt.text(20,-20,'Hello WOrld')
    for i in xrange(0,len(dataset.fileNames)):
        #load image
        name = dataset.fileNames[i].split('.')[0]
        name = name+suffix
        im = plt.imread(path+name)
        plt.imshow(im)
        text = 'Name: '+name + ', label: '+dataset.target[i]+ ', classified as: '+dataset.classifiedAs[i]
        t.set_text(text)
        plt.draw()
        plt.waitforbuttonpress()  
    
def MouthTwo2FourComparer(target,test):
    if target==test:
        return True
    if target=='closed' or target == 'narrow':
        if test=='closed':
            return True
        else:
            return False
    if target == 'open' or target == 'wideOpen':
        if test == 'open':
            return True
        else:
            return False
        
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
        if self.dataset.hasbeenClassified:
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
        
    