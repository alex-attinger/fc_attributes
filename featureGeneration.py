
import cv2
import numpy as np
import customHog as chog
from sklearn.feature_extraction.image import extract_patches_2d
import utils

class dataContainer:
    def __init__(self,labelFileDict = None):
        self.data=[]
        self.target=[]
        self.fileNames=[]
        self.targetNum=[]
        self.classifiedAs=[]
        self.hasBeenClassified = False
        if(labelFileDict and type(labelFileDict) is dict):
            keys=labelFileDict.keys()
            for key in keys:
                self.data.append([])
                self.target.append(labelFileDict[key])
                self.fileNames.append(key)
                self.targetNum.append(None)
        elif(labelFileDict and type(labelFileDict) is list):
            for i in labelFileDict:
                self.data.append([])
                self.fileNames.append(i)
                self.target.append('')
#        else:
#            msg='unsupported type as input: ' + str(type(labelFileDict))
#            raise Exception(msg)
            
            
        
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
    def addContainer(self,Y):
        idx = len(Y.data)
        for i in range(idx):
            self.data.append(Y.data[i])
            self.target.append(Y.target[i])
            self.fileNames.append(Y.fileNames[i])
            self.targetNum.append(Y.targetNum[i])
        
def getHogFeature(dataC,roi,path=None,ending=None,extraMask = None,orientations=3,pixels_per_cell=(8,8),cells_per_block=(4,2),maskFromAlpha=False,strel=None):
    print 'generating Hog Feature'    
    if len(dataC.data)==0:
        dataC.data=['' for i in xrange(0,len(dataC.fileNames))]
    if strel is None:
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
    

    for i in xrange(0,len(dataC.fileNames)):
        f=dataC.fileNames[i]        
        if ending is not None:
            
            prefix = f.split('.')[0]
            f_name = path+prefix+ending
        else:
            f_name=path+f
            
        im = cv2.imread(f_name,-1)
        if im is None:
            m='No image found at: '+f_name
            raise Exception(m)
        
        if maskFromAlpha:
            #alpha channel == 255 for pixels with data, 0 otherwise
            #erode the alpha region to get rid of boundary effects in the diff computation in chog
            alpha=im[:,:,3]
            extraMask = (cv2.erode(alpha,strel)==0)[roi[0]:roi[1],roi[2]:roi[3]] #extra Mask: pixels to be set to 0 after diff computation
            
            
        
        if im.ndim==3:
            im = cv2.cvtColor(im,cv2.cv.CV_RGB2GRAY)
        
        if im is None:
            msg = 'did not find '+str(f_name) 
            raise Exception(msg)
        if roi is None:
            ex=im
        else:
            ex = im[roi[0]:roi[1],roi[2]:roi[3]]

        #vals = feature.hog(ex,orientations=9,pixels_per_cell=(16,16),cells_per_block=(3,3),normalise=False)
        #vals = chog.hog(ex,orientations = 5, pixels_per_cell = (16,16),cells_per_block=(16,8),normalise=True,mask = extraMask)
        #vals = chog.hog(ex,orientations = 3, pixels_per_cell = (16,16),cells_per_block=(8,4),normalise=True,mask = extraMask)
        vals = chog.hog(ex,orientations = orientations, pixels_per_cell = pixels_per_cell,cells_per_block=cells_per_block,normalise=True,mask = extraMask)

        dataC.data[i].extend(vals)
     
    print 'hog-feature Lengths: {}'.format(len(vals))
    return
    
def getPoseLabel(dataC,pathToPoseFiles):
    for i in range(len(dataC.fileNames)):
        f=dataC.fileNames[i]
        f=f.split('.')[0]
        f=f+'.frog'
        pose = utils.parseSingleLabelFile(pathToPoseFiles+f,'rotY')
        pose = np.double(pose)*180/np.pi
        dataC.data[i].append(pose)
    return
    
def getColorHistogram(dataC,roi,path=None,ending=None,colorspace=None,bins = 9,range=(1.0,255.0)):
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
        c=np.shape(im)
        if len(c) <3 or c[2]<3:
            msg = str(f_name)+'does not have three color channels'
            print msg
            im = cv2.cvtColor(im,cv2.cv.CV_GRAY2RGB)
        
        if colorspace is not None:
            if colorspace in['hsv','HSV']:
                im = cv2.cvtColor(im,cv2.cv.CV_RGB2HSV)
            if colorspace in ['hls','HLS']:
                im = cv2.cvtColor(im,cv2.cv.CV_RGB2HLS)
            if colorspace in ['lab','LAB','Lab']:
                im = cv2.cvtColor(im,cv2.cv.CV_RGB2Lab)
        
        if roi is None:
            ex=im
        else:
            ex = im[roi[0]:roi[1],roi[2]:roi[3]]
        for j in xrange(0,3):
            vals=np.histogram(ex[:,:,j],bins=bins,range=range)[0]
            if not np.shape(vals)==(bins,):
                msg = f_name +'has wrong hist shape' + str(np.shape(vals))
                raise Exception(msg)
            dataC.data[i].extend(vals)
     
    print 'feature length: {}'.format(len(dataC.data[0]))
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
    
def getAllImages(path,fileNames,imsize,roi=None):
    n=len(fileNames)
    if roi is None:
        sh=(imsize[0],imsize[1],n)
    else:
        sh=(roi[1]-roi[0],roi[3]-roi[2],n)
    X=np.zeros(sh)
    for i in range(n):
        full=path+fileNames[i]
        im=cv2.imread(full,-1)
        im = cv2.cvtColor(im,cv2.cv.CV_RGBA2GRAY)
        if roi is not None:
            ex = im[roi[0]:roi[1],roi[2]:roi[3]]
            X[:,:,i]=ex.copy()
        else:
            X[:,:,i]=im.copy()
            
    return X
    
def getImagePatchStat(dataC,path=None,ending=None,patchSize = None,overlap = 0,mode=0):
    if len(dataC.data)==0:
        dataC.data=['' for i in xrange(0,len(dataC.fileNames))]
        
    f=dataC.fileNames[12]        
    
    if ending is not None:
            
        prefix = f.split('.')[0]
        f_name = path+prefix+ending
    else:
        f_name=path+f
            
    im = cv2.imread(f_name,-1)
    c=np.shape(im)
    nRows = c[0]-patchSize[0]+1
    nCols = c[1]-patchSize[1]+1
    
    for i in xrange(0,len(dataC.fileNames)):
        f=dataC.fileNames[i]        
        if ending is not None:
            
            prefix = f.split('.')[0]
            f_name = path+prefix+ending
        else:
            f_name=path+f
            
        im = cv2.imread(f_name,-1)
        c=np.shape(im)
        if len(c) ==3:
            msg = str(f_name)+'only grayscale patches'
            print msg
            im = cv2.cvtColor(im,cv2.cv.CV_RGB2GRAY)
            
        p=extract_patches_2d(im,patchSize)
        for j in range(0,p.shape[0]):
            r=j//nCols
            c=j%nCols
            if r%(patchSize[0]-overlap)== 0 and c%(patchSize[1]-overlap)==0:            
                ex=p[j]
                if mode == 0:
                    dataC.data[i].extend([ex.mean(), ex.std()])
                if mode == 1:
                    dataC.data[i].extend([ex.mean()])
                if mode == 2:
                    dataC.data[i].extend([ex.std()])
        
       
     
    print 'feature length: {}'.format(len(dataC.data[0]))
    return
    
def getAllImagesFlat(path,fileNames,imsize,roi=None,resizeFactor = 1.):
    n=len(fileNames)
    if roi is None:
        sh=(n,imsize[0]*imsize[1]*resizeFactor*resizeFactor)
    else:
        sh=(n,(roi[1]-roi[0])*(roi[3]-roi[2]))
    X=np.zeros(sh)
    
    for i in range(n):
        try:
            full=path+fileNames[i]
            im=cv2.imread(full,-1)
            if resizeFactor != 1.0:
                shape=(int(imsize[0]*resizeFactor),int(imsize[1]*resizeFactor))
                im=cv2.resize(im,shape)
            if im.ndim>2:
                im = cv2.cvtColor(im,cv2.cv.CV_RGBA2GRAY)
            if roi is not None:
                ex = im[roi[0]:roi[1],roi[2]:roi[3]]
                X[i,:]=ex.flatten()
            else:
                X[i,:]=im.flatten()
        except Exception as e:
            print 'Error '+ full
            raise e
            
    return X

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