import cv2
import numpy as np
import winsound
import time

class getTemplate:
    FREQ = 300
    DURATION = 250
    def getImage(self):
        vc = cv2.VideoCapture(0)
        if vc.isOpened(): # try to get the first frame
            rval, frame = vc.read()
        else:
            rval = False 
        print rval
        rval, frame = vc.read()
        h,w,d = frame.shape
        board = np.float32([ [w*3/8,h*1/8],[w*3/8,h*7/8],[w*5/8,h*7/8],[w*5/8,h*1/8],\
                [w*1/8,h*3/8],[w*7/8,h*3/8],[w*1/8,h*5/8],[w*7/8,h*5/8]]).reshape(-1,1,2)    
        start = time.time()
        while(rval and time.time() - start < 8):
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
        
      
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        imgSplit = cv2.split(frame)
        gray = cv2.GaussianBlur(gray,(5,5),0)
        thres = cv2.adaptiveThreshold(gray,255,1,1,11,2) 
        vc.release() 
        cv2.imshow('preview', thres)
        key = cv2.waitKey(5)
        time.sleep(5.5)  
        return frame
        
        


