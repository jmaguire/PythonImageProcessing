import cv2
import numpy as np
import getTemplate as gt
import time
from kmeans import tictactoeMeans

PAD = 15
## Image/Dectector Class
class CapturedImage:
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

def getCircle(radius,width):
    degrees = [x for x in xrange(0,360,2)]

    circle = []
    for degree in degrees:
        radian = np.pi / 180. * degree
        for i in range(width): ## 4 wide circle
            x = (radius-i)*np.cos(radian)
            y = (radius-i)*np.sin(radian)
            circle.append([x,y])
    return np.array(circle);


    
## Get Matches. Return empty array if no descriptors are found
def getMatches(captured, target):
    ##initialize good matches
    goodMatches = []

    ## Create Flann Tree
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    ## Check edge cases
    if (captured.getDescriptors() is None or target.getDescriptors() is None):
        return goodMatches
    
    if (len(captured.getDescriptors()) <= 4 or len(target.getDescriptors()) <= 0):
        return goodMatches
    
    matches = flann.knnMatch(target.getDescriptors(),captured.getDescriptors(),k=2)
    # store all the good matches as per Lowe's ratio test.
    
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            goodMatches.append(m)
    return goodMatches

def extractQuadrant(frame,quadrant):
    lowerRight = None
    if quadrant == 1:
        lowerRight = np.array([xmin,ymin])-PAD
        upperLeft = np.array([w*1/8,h*1/8])
    elif quadrant == 2:
        lowerRight = np.array([xmax,ymin])-PAD
        upperLeft = np.array([xmin+PAD,h*1/8])
    elif quadrant == 3:
        lowerRight = np.array([w*7/8,ymin])
        upperLeft = np.array([xmax+PAD,h*1/8])
    elif quadrant == 4:
        lowerRight = np.array([xmin,ymax])-PAD
        upperLeft = np.array([w*1/8,ymin+PAD])
    elif quadrant == 5:
        lowerRight = np.array([xmax,ymax])- PAD
        upperLeft = np.array([xmin,ymin]) + PAD
    elif quadrant == 6:    
        lowerRight = np.array([w*7/8,ymax - PAD])
        upperLeft = np.array([xmax,ymin]) + PAD
    elif quadrant == 7:    
        lowerRight = np.array([xmin-PAD,h*7/8])
        upperLeft = np.array([w*1/8,ymax]) + PAD
    elif quadrant == 8:    
        lowerRight = np.array([xmax-PAD,h*7/8])
        upperLeft = np.array([xmin,ymax]) + PAD
    elif quadrant == 9:
        lowerRight = np.array([w*7/8,h*7/8])
        upperLeft = np.array([xmax,ymax]) + PAD
    return frame[upperLeft[0]:lowerRight[0],upperLeft[1]:lowerRight[1]]

def circleAtQuadrant(circle, quadrant,xlims,ylims,w,h):
    
    ##ALWAYS USE NP.COPY IF YOU USE THIS TO CALC A NEW CIRCLE
    newCircle = np.copy(circle)
    x ,y  = 0 , 0
    if quadrant == 1:
        x = xlims[0]/2 + PAD 
        y = ylims[0]/2
    if quadrant == 2:
        x = (xlims[0]+xlims[1])/2
        y = ylims[0]/2
    if quadrant == 3:
        x = (w + xlims[1])/2 - PAD 
        y = ylims[0]/2
    if quadrant == 4:
        x = xlims[0]/2 + PAD  
        y = (ylims[0]+ylims[1])/2
    if quadrant == 5:
        x = (xlims[0]+xlims[1])/2
        y = (ylims[0]+ylims[1])/2
    if quadrant == 6:
        x = (w + xlims[1])/2 - PAD 
        y = (ylims[0]+ylims[1])/2
    if quadrant == 7:
        x = xlims[0]/2 + PAD 
        y = (ylims[1] + h)/2
    if quadrant == 8:
        x = (xlims[0]+xlims[1])/2
        y = (ylims[1] + h)/2
    if quadrant == 9:
        x = (w + xlims[1])/2 - PAD 
        y = (ylims[1] + h)/2
    newCircle[:,0] = circle[:,0] + x
    newCircle[:,1] = newCircle[:,1] + y 
    return newCircle


## Initialize detectors  
##--------------------------------------------------------------
##--------------------------------------------------------------

## Initialize Surf Detector
detector = cv2.SURF()

## Get target and extract keypoints/descriptors
o = gt.getTemplate()
centroidsTarget, midpointsTarget, frame = o.getImage(1)     
image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
keypoints,descriptors = detector.detectAndCompute(image,None)   
target = CapturedImage(image, descriptors,keypoints)


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
##--------------------------------------------------------------
##--------------------------------------------------------------
    
    
    
## BOARD AND INITIAL VARIABLES
##--------------------------------------------------------------
##--------------------------------------------------------------

# h,w = target.getImage().shape


# print h,w

# xmin = min(centroidsTarget[:,0])
# xmax = max(centroidsTarget[:,0])
# ymin = min(centroidsTarget[:,1])
# ymax = max(centroidsTarget[:,1])

# h,w,d = frame.shape

## Doesn't really work because it needs to board to be just tic tac toe,
## However this doesn't give us enough features
# board = np.float32([ [xmin,h*1/8],[xmin,h*7/8],[xmax,h*7/8],[xmax,h*1/8],\
        # [w*1/8,ymin],[w*7/8,ymin],[w*1/8,ymax],[w*7/8,ymax]]).reshape(-1,1,2)   

boardX = [w*3/8,w*5/8]
boardY = [h*3/8,h*5/8]
board = np.float32([ [boardX[0],h*1/8],[boardX[0],h*7/8],[boardX[1],h*7/8],[boardX[1],h*1/8],\
    [w*1/8,boardY[0]],[w*7/8,boardY[0]],[w*1/8,boardY[1]],[w*7/8,boardY[1]]]).reshape(-1,1,2)    

## Outline of original image
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

## Homography
H = None

## Statistics to measure blockage
norm, oldNorm, oldBoard = None, None, None;
avergingPeriod = 50
avgDeltNorm = np.array([-1]*avergingPeriod);
blocked = False
counter = 0

## Standard Circle 
radius = (boardY[1] - boardY[0] - 2*PAD)/2

##ALWAYS USE NP.COPY IF YOU USE THIS TO CALC A NEW CIRCLE
circle = getCircle(radius, 5)

## Pieces
pieces = [None]*9

## Add two Circles
pieces[1] = 'O'
pieces[5] = 'O'
##--------------------------------------------------------------
##--------------------------------------------------------------

while True:
    ## Get Matches
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    keypoints,descriptors = detector.detectAndCompute(im,None)
    captured = CapturedImage(im, descriptors,keypoints)
    
    matches = getMatches(captured, target)
    
    ## Only update Homography and board if we can see it
    if len(matches) > 4:
        ## Get points, find Homography
        src_pts = np.float32([ target.getKeypoints()[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ captured.getKeypoints()[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,6.0)

        matchesMask = mask.ravel().tolist()
        
        ## Check for X's
        
        ## update board

    ## draw board. 
    if H is not None: 
        dst = cv2.perspectiveTransform(pts,H)
        color = [255,0,0]
        ## Draw tictactoe board
        newBoard = np.int32(cv2.perspectiveTransform(board,H));
        
        # if oldBoard != None:
            # norm = np.sum(np.apply_along_axis(np.linalg.norm,2,newBoard - oldBoard)) 
        
        # if oldNorm != None:
            # ## collect samples of difference in norms
            # if counter <= avergingPeriod:
                # avgDeltNorm[counter % len(avgDeltNorm)] = norm-oldNorm
            # else:
                # if abs(norm-oldNorm) > np.mean(avgDeltNorm) + .5*np.std(avgDeltNorm):
                    # print 'yikes',counter
                    # blocked = True
                # else:
                    # blocked = False
        
        # if blocked:
            # print 'here'
            # color = [0,255,0]
            # newBoard = oldBoard
            
        ## Draw Pieces

        for quadrant, piece in enumerate(pieces):
            if piece == 'O':
                circleQuad = circleAtQuadrant(circle, quadrant, boardX,boardY,w,h);
                ## NOTE we have to pack 2D arrays in another array for perpecticeTransform 
                ## to work
                circleQuad = np.array([circleQuad])
                ## Unpack
                circleQuad = cv2.perspectiveTransform(circleQuad,H)[0];
                for point in circleQuad.astype(int):
                    cv2.circle(frame,tuple(point),2,(255,0,0))
        ## Draw Board Lines          
        for i in [0,2,4,6]:
            ## end points of lines
            pt1 = tuple(newBoard[i][0])
            pt2 = tuple(newBoard[i+1][0])
            cv2.line(frame,pt1,pt2,color,thickness = 2)
        # for m in matches:
            # center = (int(keypoints[m.trainIdx].pt[0]),int(keypoints[m.trainIdx].pt[1]))
            # cv2.circle(frame,center,10,(255,0,0))
        img2 = cv2.polylines(frame,[np.int32(dst)],True,255,3)
    
    oldBoard = newBoard
    oldNorm = norm

    cv2.imshow('preview', frame)
    rval, frame = vc.read()
    counter += 1
    if cv2.waitKey(5) == 27: # exit on ESC
        break

cv2.destroyWindow("preview")


