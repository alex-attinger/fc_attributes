# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:06:12 2013

@author: attale00
"""
import matplotlib.pyplot as plt
import numpy as np
import utils
import cv2


def coloredProbabilityDistribution(data):
    '''
    data: list of tuples with t[0]: label
                                t[1]:probability
                                
    '''
    data = sorted(data,lambda x,y: cmp(y[1],x[1]))    
    prob = [p[1] for p in data]
    labels = [p[0] for p in data ]
    c=['r' if i==0 else 'b' for i in labels]
    s=[10 for i in labels]
    for i in range(1,len(prob)-1):
        if c[i-1]== c[i+1] and c[i-1]!=c[i]:
            s[i]=20
    plt.figure()
    plt.title('Sorted Probabilities')
    plt.xlabel('Sample number')
    plt.ylabel('Probability of Class 1')
    plt.grid()
    plt.scatter(range(0,len(prob)),prob,c=c,marker=',',edgecolors='None',s=s,alpha=.5)    
    
    

def showICAComponents(components,size,nR,nC):
    plt.figure()
    counter = 1
    for r in range(nR):
        for c in range(nC):
            plt.subplot(nR,nC,counter)
            counter+=1
            plt.imshow(components[counter-1,:].reshape(size[0],size[1]), cmap="gray")
            plt.axis("off")
    
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
    
    

def plotExcerpt(path='/local/attale00/GoodPose/extracted_alpha/grayScale',roi=(0,250,60,452),showDiff=''):


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
        cv2.rectangle(big,(roi[2],roi[1]),(roi[3],roi[0]),255,2)
        excerpt = im[roi[0]:roi[1],roi[2]:roi[3]]
    
        if showDiff in ['y','Y']:
            excerpt = np.diff(excerpt,n=1,axis=0)
        if showDiff in ['X','x']:
            excerpt = np.diff(excerpt,n=1,axis=1)
        big[roi[0]:roi[1],sh[1]+roi[2]:sh[1]+roi[3]]=excerpt
        cv2.imshow('Panorama',big)
        k=cv2.waitKey(0)
        if k==27:
            break
    cv2.destroyAllWindows()

    print('done')
    
def plotPoses(posesdic):
        
        totals,per,labels = _sortPoseforbarplot(posesdic)
        
        plt.figure()
        plt.bar(range(len(totals)),per)

        plt.xticks(range(len(totals)),labels)
        plt.show()
    
def _sortPoseforbarplot(posesdic):
    p={'140':'-15', '080':'-45', '190':'45', '130':'-30', '050':'15', '051':'0', '041':'30'}
    l=['080','130','140','051','050','041','190']
    totals={}
    per=[]
    for i in l:
        n=sum(posesdic[i])
        totals[p[i]]=n
        per.append(posesdic[i][1]*1./n)
    labels =[p[i] for i in l]
    labels = [i +'\n n={}'.format(totals[i]) for i in labels ]
    return (totals,per,labels)

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
            text = 'Name: '+name + ', label: '+self.dataset.target[self.counter]+ ', classified as: {}'.format(self.dataset.classifiedAs[self.counter])
            text +='\n Probabilities: {} {}'.format(self.dataset.probabilities[self.counter][0],self.dataset.probabilities[self.counter][1])
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
        
   
            