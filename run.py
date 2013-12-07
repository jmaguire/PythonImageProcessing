import cv2
import numpy as np
import getTemplate as gt
import time
from game import TicTacToeState
from minimax import MinimaxAgent

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
    
def extractQuadrant(frame,quadrant,xlims,ylims,w,h):
    lowerRight = None
    if quadrant == 0:
        lowerRight = np.array([xlims[0],ylims[0]])-PAD
        upperLeft = np.array([w*1/8,h*1/8])
    elif quadrant == 1:
        lowerRight = np.array([xlims[1],ylims[0]])-PAD
        upperLeft = np.array([xlims[0]+PAD,h*1/8])
    elif quadrant == 2:
        lowerRight = np.array([w*7/8,ylims[0]-PAD])
        upperLeft = np.array([xlims[1]+PAD,h*1/8])
    elif quadrant == 3:
        lowerRight = np.array([xlims[0],ylims[1]])-PAD
        upperLeft = np.array([w*1/8,ylims[0]+PAD])
    elif quadrant == 4:
        lowerRight = np.array([xlims[1],ylims[1]])- PAD
        upperLeft = np.array([xlims[0],ylims[0]]) + PAD
    elif quadrant == 5:    
        lowerRight = np.array([w*7/8,ylims[1] - PAD])
        upperLeft = np.array([xlims[1],ylims[0]]) + PAD
    elif quadrant == 6:    
        lowerRight = np.array([xlims[0]-PAD,h*7/8])
        upperLeft = np.array([w*1/8,ylims[1] + PAD])
    elif quadrant == 7:    
        lowerRight = np.array([xlims[1]-PAD,h*7/8])
        upperLeft = np.array([xlims[0],ylims[1]]) + PAD
    elif quadrant == 8:
        lowerRight = np.array([w*7/8,h*7/8])
        upperLeft = np.array([xlims[1],ylims[1]]) + PAD

    return frame[upperLeft[1]:lowerRight[1],upperLeft[0]:lowerRight[0]], upperLeft, lowerRight

def circleAtQuadrant(circle, quadrant,xlims,ylims,w,h):
    
    ##ALWAYS USE NP.COPY IF YOU USE THIS TO CALC A NEW CIRCLE
    newCircle = np.copy(circle)
    x ,y  = 0 , 0
    if quadrant == 0:
        x = xlims[0]/2 + PAD 
        y = ylims[0]/2
    if quadrant == 1:
        x = (xlims[0]+xlims[1])/2
        y = ylims[0]/2
    if quadrant == 2:
        x = (w + xlims[1])/2 - PAD 
        y = ylims[0]/2
    if quadrant == 3:
        x = xlims[0]/2 + PAD  
        y = (ylims[0]+ylims[1])/2
    if quadrant == 4:
        x = (xlims[0]+xlims[1])/2
        y = (ylims[0]+ylims[1])/2
    if quadrant == 5:
        x = (w + xlims[1])/2 - PAD 
        y = (ylims[0]+ylims[1])/2
    if quadrant == 6:
        x = xlims[0]/2 + PAD 
        y = (ylims[1] + h)/2
    if quadrant == 7:
        x = (xlims[0]+xlims[1])/2
        y = (ylims[1] + h)/2
    if quadrant == 8:
        x = (w + xlims[1])/2 - PAD 
        y = (ylims[1] + h)/2
    newCircle[:,0] = circle[:,0] + x
    newCircle[:,1] = newCircle[:,1] + y 
    return newCircle


def getNewPieces(pieces, playerMove):
    
    ## Get XY posistion of move. TicTacToeState uses (x,y) coordinates
    def getXY(index):
        if index == 0: return [0,0]
        if index == 1: return [0,1]
        if index == 2: return [0,2]
        if index == 3: return [1,0]
        if index == 4: return [1,1]
        if index == 5: return [1,2]
        if index == 6: return [2,0]
        if index == 7: return [2,1]
        if index == 8: return [2,2]
    ## Build Board
    board = [[None]*3 for j in range(3)]
    for i, piece in enumerate(pieces):
        row, col = getXY(i)
        board[row][col] = piece
        
    ## Player Move
    action = getXY(playerMove)
    gameState = TicTacToeState(board,'x')
    gameState = gameState.generateSuccessor(action)
    if None in pieces:
        ## Computer Move
        player = MinimaxAgent()
        action = player.getAction(gameState)
        if action != None:
            gameState = gameState.generateSuccessor(action)
    return gameState.board[0] + gameState.board[1] + gameState.board[2]
    
## Initialize detectors  
##--------------------------------------------------------------
##--------------------------------------------------------------

## Initialize Surf Detector
detector = cv2.SURF(85)

## Get target and extract keypoints/descriptors
o = gt.getTemplate()
frame = o.getImage(10)     
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

h,w,d = frame.shape
boardX = [w*3/8,w*5/8]
boardY = [h*3/8,h*5/8]
board = np.float32([ [boardX[0],h*1/8],[boardX[0],h*7/8],[boardX[1],h*7/8],[boardX[1],h*1/8],\
    [w*1/8,boardY[0]],[w*7/8,boardY[0]],[w*1/8,boardY[1]],[w*7/8,boardY[1]]]).reshape(-1,1,2)    

## Outline of original image
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

## X-template to use for matching
X_template = cv2.imread('X_rots.jpg',0)
X_w, X_h = X_template.shape[::-1]

## Homography
H = None

## Statistics to measure blockage
norm, oldNorm, oldBoard, H_old = None, None, None, None;
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

# filter array
xcounts = [0]*9
##--------------------------------------------------------------
##--------------------------------------------------------------

cv2.namedWindow("Rectified")

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

        successInv, invH = cv2.invert(H)
        imRect = cv2.warpPerspective(im, invH, (w,h))

        matchesMask = mask.ravel().tolist()
        
        xPieces = [None]*9
        if None in pieces:
            ## Check for X's
            for quadrant in xrange(0,9):

                quadrantImage, upperLeft, lowerRight = extractQuadrant(imRect, quadrant, boardX, boardY, w, h)
                # cv2.rectangle(frame,(top_left[0], top_left[1]), (bottom_right[0], bottom_right[1]), 100, 2)
                res = cv2.matchTemplate(quadrantImage,X_template,cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) # this currently searches whole image.  can use mask for each box

                # print "maxval = ", max_locval
                
                if max_val >= 0.4: # 0.4 detemined empirically based on my board
                    # print "Found an X in quadrant ", quadrant, "value = ", max_val, "loc ", max_loc
                    top_left = max_loc + upperLeft
                    top_left = (top_left[0], top_left[1])
                    bottom_right = (top_left[0] + X_w, top_left[1] + X_h)
                    
                    cv2.rectangle(imRect,top_left, bottom_right, 100, 2) # specific box
                    cv2.rectangle(imRect,(upperLeft[0], upperLeft[1]), (lowerRight[0], lowerRight[1]), 200, 2) #quadrant box
                
                    xPieces[quadrant] = 'x'

            
            # update board
            for quadrant in xrange(0,9):
                if (pieces[quadrant] != 'x'):
                    if (xPieces[quadrant] == 'x'):
                        xcounts[quadrant] += 1
                        if xcounts[quadrant] == 3:
                            pieces = getNewPieces(pieces, quadrant)
                    else:
                        xcounts[quadrant] = 0

    ## draw board. 
    if H is not None: 
        dst = cv2.perspectiveTransform(pts,H)
        color = [255,0,0]
        newBoard = np.int32(cv2.perspectiveTransform(board,H));
        if H_old != None:
            #norm = np.sum(np.apply_along_axis(np.linalg.norm,2,newBoard - oldBoard)) 
            norm = np.linalg.norm(H- H_old, 2)

        
        if oldNorm != None:
            ## collect samples of difference in norms
            if counter <= avergingPeriod:
                avgDeltNorm[counter % len(avgDeltNorm)] = norm-oldNorm
            else:
                if abs(norm-oldNorm) > np.mean(avgDeltNorm) + 1.5*np.std(avgDeltNorm):
                    blocked = True
                else:
                    blocked = False
        
        if blocked:
            color = [0,255,0]
            ##newBoard = oldBoard
            H = H_old;
            ## redraw
            newBoard = np.int32(cv2.perspectiveTransform(board,H));
            dst = cv2.perspectiveTransform(pts,H)
            
        ## Draw 
        ## Draw tictactoe board
        
        for quadrant, piece in enumerate(pieces):
            if piece == 'o':
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
        H_old = H

        cv2.imshow('Rectified', imRect)
        
    cv2.imshow('preview', frame)
    rval, frame = vc.read()
    counter += 1
    if cv2.waitKey(5) == 27: # exit on ESC
        break

cv2.destroyWindow("preview")
cv2.destroyWindow("Rectified")


