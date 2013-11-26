import cv2
import numpy as np
import getTemplate as gt
import time
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


## Initialize Surf Detector
detector = cv2.SURF()

## Get target and extract keypoints/descriptors
o = gt.getTemplate()
centroids, midpointsTarget, frame = o.getImage(1)     
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

def getMidpoints(frame):
    h,w,d = frame.shape
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    lines = cv2.HoughLines(edges,1,np.pi/180,100)
    midpoints = []
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        midpointX = int((x2+x1)/2.0)
        midpointY = int((y1+y2)/2.0)
        
        if midpointX <= 20:
            midpointX = int(w/2)
        if midpointY <= 20:
            midpointY = int(h/2)
        midpoint = (midpointX,midpointY)
        midpoints.append(midpoint)
        cv2.circle(frame,midpoint,10,(255,0,0))
    cv2.imshow('preview', frame)
    key = cv2.waitKey(5)
    #time.sleep(10)  
    return np.int32(midpoints).reshape(-1,1,2)
    
##Build stuff. TEMPORARY. Should extract board from target
h,w = target.getImage().shape
print h,w

xmin = min(centroids[:,0])
xmax = max(centroids[:,0])
ymin = min(centroids[:,1])
ymax = max(centroids[:,1])

h,w,d = frame.shape
board = np.float32([ [xmin,h*1/8],[xmin,h*7/8],[xmax,h*7/8],[xmax,h*1/8],\
        [w*1/8,ymin],[w*7/8,ymin],[w*1/8,ymax],[w*7/8,ymax]]).reshape(-1,1,2)   


# board = np.float32([ [w*3/8,h*1/8],[w*3/8,h*7/8],[w*5/8,h*7/8],[w*5/8,h*1/8],\
    # [w*1/8,h*3/8],[w*7/8,h*3/8],[w*1/8,h*5/8],[w*7/8,h*5/8]]).reshape(-1,1,2)    

## Outline of original image
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

## Homography
H = None
norm, oldNorm, oldBoard = None, None, None;
avergingPeriod = 50
avgDeltNorm = np.array([-1]*avergingPeriod);
blocked = False
counter = 0


while True:
    ## Get Matches
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    midpts = getMidpoints(frame)
    for i in [0,2,4,6]:
        ## end points of lines
        pt1 = tuple(board[i][0])
        pt2 = tuple(board[i+1][0])
        cv2.line(frame,pt1,pt2,(0,255,255),thickness = 2)
    cv2.imshow('preview', frame)
    key = cv2.waitKey(5)
    time.sleep(1)  
    # keypoints,descriptors = detector.detectAndCompute(im,None)
    # captured = CapturedImage(im, descriptors,keypoints)
    
    # matches = getMatches(captured, target)
    
    # ## Only update Homography and board if we can see it
    # if len(matches) > 4:
        # ## Get points, find Homography
        # src_pts = np.float32([ target.getKeypoints()[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        # dst_pts = np.float32([ captured.getKeypoints()[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
        # H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,6.0)
        # matchesMask = mask.ravel().tolist()
        
        # ## Check for X's
        
        # ## update board

    # ## draw board. 
    # if H is not None: 
        # dst = cv2.perspectiveTransform(pts,H)
        # color = [255,0,0]
        # ## Draw tictactoe board
        # newBoard = np.int32(cv2.perspectiveTransform(board,H));
        
        # # if oldBoard != None:
            # # norm = np.sum(np.apply_along_axis(np.linalg.norm,2,newBoard - oldBoard)) 
        
        # # if oldNorm != None:
            # # ## collect samples of difference in norms
            # # if counter <= avergingPeriod:
                # # avgDeltNorm[counter % len(avgDeltNorm)] = norm-oldNorm
            # # else:
                # # if abs(norm-oldNorm) > np.mean(avgDeltNorm) + .5*np.std(avgDeltNorm):
                    # # print 'yikes',counter
                    # # blocked = True
                # # else:
                    # # blocked = False
        
        # # if blocked:
            # # print 'here'
            # # color = [0,255,0]
            # # newBoard = oldBoard
     
        # for i in [0,2,4,6]:
            # ## end points of lines
            # pt1 = tuple(newBoard[i][0])
            # pt2 = tuple(newBoard[i+1][0])
            # cv2.line(frame,pt1,pt2,color,thickness = 2)
        # # for m in matches:
            # # center = (int(keypoints[m.trainIdx].pt[0]),int(keypoints[m.trainIdx].pt[1]))
            # # cv2.circle(frame,center,10,(255,0,0))
        # img2 = cv2.polylines(frame,[np.int32(dst)],True,255,3)
    
    # oldBoard = newBoard
    # oldNorm = norm

    # cv2.imshow('preview', frame)
    rval, frame = vc.read()
    counter += 1
    if cv2.waitKey(5) == 27: # exit on ESC
        break

cv2.destroyWindow("preview")


