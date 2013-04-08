
import cv2
import numpy as np
import skimage.feature as feature
import customHog as chog

class dataContainer:
    def __init__(self,labelFileDict = None):
        self.data=[]
        self.target=[]
        self.fileNames=[]
        self.targetNum=[]
        self.classifiedAs=[]
        if(labelFileDict):
            keys=labelFileDict.keys()
            for key in keys:
                self.data.append([])
                self.target.append(labelFileDict[key])
                self.fileNames.append(key)
            
            
        
    def splitInTestAndTraining(self, frac = .6):
        outTest = dataContainer()
        outTraining = dataContainer()
        stopInd = int(frac*len(self.data))
        idx = np.arange(0,len(self.data))
        np.random.shuffle(idx)
        for i in xrange(0,stopInd):
            outTraining.data.append(self.data[idx[i]])
            outTraining.target.append(self.target[idx[i]])
            outTraining.fileNames.append(self.fileNames[idx[i]])
            outTraining.targetNum.append(self.targetNum[idx[i]])
            
        for i in xrange(stopInd,len(self.data)):
            outTest.data.append(self.data[idx[i]])
            outTest.target.append(self.target[idx[i]])
            outTest.fileNames.append(self.fileNames[idx[i]])
            outTest.targetNum.append(self.targetNum[idx[i]])
        return (outTraining,outTest)
        
def getHogFeature(dataC,roi,path=None,ending=None,extraMask = None):
    if len(dataC.data)==0:
        dataC.data=['' for i in xrange(0,len(dataC.fileNames))]

    for i in xrange(0,len(dataC.fileNames)):
        f=dataC.fileNames[i]        
        if ending is not None:
            
            prefix = f.split('.')[0]
            f_name = path+prefix+ending
        else:
            f_name=path+f
            
        im = cv2.imread(f_name,-1)
  
        ex = im[roi[0]:roi[1],roi[2]:roi[3]]

        #vals = feature.hog(ex,orientations=9,pixels_per_cell=(16,16),cells_per_block=(3,3),normalise=False)
        vals = chog.hog(ex,orientations = 5, pixels_per_cell = (16,16),cells_per_block=(16,8),normalise=True,mask = extraMask)
        dataC.data[i].extend(vals)
     

    return

def getPixelValues(dataC,roi,path=None,ending=None, mask = None,scaleFactor = 4):
    
     for i in xrange(0,len(dataC.fileNames)):
        f=dataC.fileNames[i]        
        if ending:
            
            prefix = f.split('.')[0]
            f_name = path+prefix+ending
        else:
            f_name=path+f
            
        im = cv2.imread(f_name,-1)
        s= np.shape(im)
        if len(s)>2 and s[2] == 4:
            im = im[:,:,0:2]
            
        ex = im[roi[0]:roi[1],roi[2]:roi[3]]
        shape = (int((roi[1]-roi[0])/scaleFactor),int((roi[3]-roi[2])/scaleFactor))
        ex = cv2.resize(ex,shape)
        if mask is not None:
            mask = cv2.resize(np.uint8(mask),shape)
            mask = mask>0
            ex = ex[mask]
            
        ex=np.sqrt(ex)
        vals = ex.flatten()
        dataC.data[i].extend(vals)

        

def getHistogram(nbins,roi,dataC=None,hrange=(1.0,255,0),path=None,ending=None):
    '''
    nbins: number of bins for each histogram
    roi: tuple with the content (rmin,rmax,cmin,cmax)
    labelFileDict: dictionary containing filename-label pairs, None for unlabeled data
    path: path to folder containing images
    ending: optional alternative ending
    
    '''
    
    if len(dataC.data)==0:
        dataC.data=['' for i in xrange(0,len(dataC.fileNames))]
    
    
    
    
    for i in xrange(0,len(dataC.fileNames)):
        f=dataC.fileNames[i]        
        if ending:
            
            prefix = f.split('.')[0]
            f_name = path+prefix+ending
        else:
            f_name=path+f
            
        im = cv2.imread(f_name,-1)
  
        ex = im[roi[0]:roi[1],roi[2]:roi[3]]

        vals,bins = np.histogram(ex,bins=nbins,range=hrange)
       
        dataC.data[i].extend(vals)
     

    return 

def getMeanAndVariance(roi,dataC,path=None,ending=None,extraMask = None,picSize = (512,512)):
    '''

    roi: tuple with the content (rmin,rmax,cmin,cmax)
    labelFileDict: dictionary containing filename-label pairs, None for unlabeled data
    path: path to folder containing images
    ending: optional alternative ending
    
    '''
    if len(dataC.data)==0:
        dataC.data=['' for i in xrange(0,len(dataC.fileNames))]
    
    roi_mask = np.zeros(picSize,dtype=np.bool)
    roi_mask[roi[0]:roi[1],roi[2]:roi[3]]=True
    
    if extraMask is None:
        compoundMask = roi_mask
    else:
        compoundMask = roi_mask & extraMask
    
    
    for i in xrange(0,len(dataC.fileNames)):
        try:
            f=dataC.fileNames[i]        
            if ending is not None:
            
                prefix = f.split('.')[0]
                f_name = path+prefix+ending
            else:
                f_name=path+f
                
            im = cv2.imread(f_name,-1)
            red = im[:,:,0][compoundMask]
            green = im[:,:,1][compoundMask]
            blue = im[:,:,2][compoundMask]            
           
            mean = [red.mean(),green.mean(),blue.mean()]
            covMat = np.cov([red,green,blue])
            dataC.data[i].extend(mean)
            dataC.data[i].extend(covMat.flatten())
        
        except Exception as e:
            print f_name
            print (e.message)
            

    return