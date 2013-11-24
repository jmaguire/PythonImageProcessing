import cv2
import numpy as np
import getTemplate as gt

## Image/Dectector Class
class Frame:
    def __init__(self,image,descriptors,keypoints):
        self.image = image
        self.descriptors = descriptors
        self.keypoints = keypoints
    def getImage(self):
        return self.image
    def getDescriptors(self):
        return self.descriptors
    def getKeypoints(self):
        return self.keypoints


## Initialize Surf Detector
detector = cv2.SURF()

## Get target and extract keypoints/descriptors
o = gt.getTemplate()
frame = o.getImage(4)     
image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
keypoints,descriptors = detector.detectAndCompute(image,None)   
target = Frame(image, descriptorsTarget,keypointsTarget)
print target.getImage() == image,target.getDescriptors() == descriptorsTarget, target.getKeypoints() == keypointsTarget
##Initialize webcam
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
rval = False
while not rval:
    rval, frame = vc.read()

if vc.isOpened(): # try to get the first frame should always work!
    rval, frame = vc.read()
else:
    rval = False 
    
    
def getMatches(descriptors, descriptorsTarget):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptorsTarget,descriptors,k=2)
    # store all the good matches as per Lowe's ratio test.
    goodMatches = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            goodMatches.append(m)
    return goodMatches



##Build stuff. TEMPORARY. Should extract board from target
h,w = target.shape
board = np.float32([ [w*3/8,h*1/8],[w*3/8,h*7/8],[w*5/8,h*7/8],[w*5/8,h*1/8],\
    [w*1/8,h*3/8],[w*7/8,h*3/8],[w*1/8,h*5/8],[w*7/8,h*5/8]]).reshape(-1,1,2)    

'''
while True:
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints,descriptors = detector.detectAndCompute(im,None)
    if descriptors is None: continue
    if len(descriptors) > 3:
        matches = getMatches(descriptors, descriptorsTarget)
        if len(matches) > 10:
            src_pts = np.float32([ keypointsTarget[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ keypoints[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            newBoard = np.int32(cv2.perspectiveTransform(board,M));
            for i in [0,2,4,6]:
                #print newBoard[i][0],newBoard[i+1][0]
                pt1 = tuple(newBoard[i][0])
                pt2 = tuple(newBoard[i+1][0])
                cv2.line(frame,pt1,pt2,[255,0,0],thickness = 2)
                
            for m in matches:
                center = (int(keypoints[m.trainIdx].pt[0]),int(keypoints[m.trainIdx].pt[1]))
                cv2.circle(frame,center,10,(255,0,0))
            img2 = cv2.polylines(frame,[np.int32(dst)],True,255,3)
        
    cv2.imshow('preview', frame)
    rval, frame = vc.read()
    key = cv2.waitKey(5)
    
    
    if key == 27: # exit on ESC
        break
'''
cv2.destroyWindow("preview")
