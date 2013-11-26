import cv2
import numpy as np
import winsound
import time
from kmeans import tictactoeMeans

class getTemplate:
    FREQ = 300
    DURATION = 250
    
    ## Extract HoughLines and Find their midpoints
    ## Also uses a modified kmeans, tictactoeMeans
    ## To extract the 4 midpoints for the 4 lines
    def getMidpoints(self, frame):
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
            x1 = int(x0 + 10000*(-b))
            y1 = int(y0 + 10000*(a))
            x2 = int(x0 - 10000*(-b))
            y2 = int(y0 - 10000*(a))
            
            #print (x1,y1),(x2,y2)
            midpointX = int((x2+x1)/2.0)
            midpointY = int((y1+y2)/2.0)
            
            if midpointX <= 20:
                midpointX = int(w/2)
            if midpointY <= 20:
                midpointY = int(h/2)
            midpoint = (midpointX,midpointY)
            midpoints.append(midpoint)
            cv2.circle(frame,midpoint,10,(255,0,0))
        matrix = np.array(midpoints)
        midpoints = np.int32(midpoints).reshape(-1,1,2);

        cv2.imshow('preview', frame)
        key = cv2.waitKey(5)

        matrix = tictactoeMeans(matrix,4,10)

        for i in range(matrix.shape[0]):
            center = (int(matrix[i,0]),int(matrix[i,1]))
            cv2.circle(frame,center,10,(255,0,255))
        cv2.imshow('preview', frame)
        key = cv2.waitKey(5)

        return matrix, midpoints
    
    ##Return 4 centroids, Midpoints, Image
    def getImage(self, duration):
    
        ## Open Capture Device
        vc = cv2.VideoCapture(0)
        if vc.isOpened(): # try to get the first frame
            rval, frame = vc.read()
        else:
            rval = False 
        print rval
        rval, frame = vc.read()
        
        ## Create Board template
        h,w,d = frame.shape
        board = np.float32([ [w*3/8,h*1/8],[w*3/8,h*7/8],[w*5/8,h*7/8],[w*5/8,h*1/8],\
                [w*1/8,h*3/8],[w*7/8,h*3/8],[w*1/8,h*5/8],[w*7/8,h*5/8]]).reshape(-1,1,2)    
       
        
        ## Display template
        start = time.time()
        while(rval and time.time() - start < duration):
            for i in [0,2,4,6]:
                #print newBoard[i][0],newBoard[i+1][0]
                pt1 = tuple(board[i][0])
                pt2 = tuple(board[i+1][0])
                cv2.line(frame,pt1,pt2,[255,0,0],thickness = 2)
            cv2.imshow('preview', frame)
            rval, frame = vc.read()
            key = cv2.waitKey(5)
        rval, frame = vc.read()
        key = cv2.waitKey(5)
        winsound.Beep(getTemplate.FREQ,getTemplate.DURATION)
        
        ## Get Midpoints, centroids
        centroids, midpoints = self.getMidpoints(frame)
        
        ## Display what was captured for 2 seconds
        cv2.imshow('preview', frame)
        key = cv2.waitKey(5)
        time.sleep(2)  
        return centroids, midpoints, frame
        
       



