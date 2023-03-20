# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 16:43:59 2023

@author: kishore prashanth
"""

import cv2
import HandTrackingModule as htm
import time
import numpy as np
#import os
#import mediapipe as mp

##################
#PARAMETERS
pTime=0 #previousTIme
cTime=0 #currTime
lmList=[] #List to store landmark id's & coordinates
xp,yp=0,0 #Initial points to draw
drawSelectColor=(102, 0, 102)
brushSize=3
##################

def preprocess():
    image = cv2.imread('D:\\SRP\\HandTrackingProject\\ROI.png')
    imgGray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(imgGray, 5)
    _,imgInv=cv2.threshold(imgGray,0,255,cv2.THRESH_BINARY_INV)
    _,imgInv1=cv2.threshold(blur,0,255,cv2.THRESH_BINARY)
    #cv2.imwrite("testing.png",imgInv1)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("testing.png",imgInv1)
    
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,8)
    #cv2.adaptiveThreshold(src,maxVal,adaptiveThresholdType,thresholdtype,blocksize,C)
    #Here threshold value is calculated by the mean of blockSize minus C => cv2.APAPTIVE_THRESHOLD_MEAN_C
    #cv2.ADAPTIVE_THRESH_GAUSSIAN_C - weighted sum (cross relation with Gaussian window) of blockSize * blockSize
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilate = cv2.dilate(thresh, kernel, iterations=6)
    #the text regions are more emphasized and larger
    #for improving the accuracy of text detection or recognition in the image.
    
    cnts,_ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    #Finding the contour that has the maximum area
    
    
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c) #x,y +> top-left coordinates, w,h => width&height
        ROI = imgInv[y:y+h, x:x+w]
        ROI=cv2.resize(ROI,(128,128))
        cv2.imwrite('ROI.png', ROI)
        break
    

cap=cv2.VideoCapture(0)
cap.set(3,960) #3-Width
cap.set(4,540) #4-Height
#Must meet aspect ratio

detector=htm.handDetector(detectionCon=0.4)

imgCanvas=np.zeros((540,960,3),np.uint8)
cpy_imgCanvas=np.zeros((540,960,3),np.uint8)


#27 - Esc
while cv2.waitKey(1)!=27:
    
    success,img=cap.read()
    img=cv2.flip(img,1) #Flip around y-axis
    
    img=detector.findHands(img,draw=False)
    lmList=detector.findPosition(img,draw=False)
    
    
    if len(lmList)!=0:
        
        #Index finger
        x1,y1=lmList[8][1],lmList[8][2]
        
        fingers=detector.fingersUp()
        #print(fingers)
        
        
        #Prepare mode:
        if fingers[0]==True and fingers[1]==True and fingers[2]==False and fingers[3]==False and fingers[4]==False:
            xp,yp=0,0
            cv2.putText(img,"PREPARE",(800,50),cv2.FONT_HERSHEY_PLAIN,2,(255, 255, 0),3)
            
            
        #WRITE mode:
        if fingers[0]==False and fingers[1] and fingers[2]==False and fingers[3]==False and fingers[4]==False:
            cv2.putText(img,"WRITE",(800,50),cv2.FONT_HERSHEY_PLAIN,2,(255, 255, 0),3)
            #cv2.circle(img,(x1,y1),8,drawSelectColor,cv2.FILLED)
            if xp==0 and yp==0:
                xp,yp=x1,y1
            cv2.line(img,(xp,yp),(x1,y1),drawSelectColor,brushSize)  
            cv2.line(imgCanvas,(xp,yp),(x1,y1),drawSelectColor,brushSize)
            xp,yp=x1,y1
            
            
        #DELETE mode:
        if fingers[0] and fingers[1]==False and fingers[2]==False and fingers[3]==False and fingers[4]:
            cv2.putText(img,"DELETE",(800,50),cv2.FONT_HERSHEY_PLAIN,2,(255, 255, 0),3)
            imgCanvas=cpy_imgCanvas.copy()
            xp,yp=0,0
            
            
        #STORE mode:
        if fingers[0]==False and fingers[1] and fingers[2] and fingers[3]==False and fingers[4]==False:
            xp,yp=0,0
            cv2.putText(img,"STORE",(800,50),cv2.FONT_HERSHEY_PLAIN,2,(255, 255, 0),3)
            cv2.imwrite("ROI.png",imgCanvas)
            preprocess()
                
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    
    cv2.putText(img,str(int(fps)),(25,50),cv2.FONT_HERSHEY_PLAIN,2,(255, 255, 0),3)
    
    
    #####EMBEDDING CANVAS BACK TO THE IMAGE#######
    
    #cv2.imshow("imgCanvas",imgCanvas)
    
    imgGray=cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    
    #Low intensity px (close to 0) remains in 0
    #High intensity px gets its corresponding shade either in white or gray
    #cv2.imshow("imgGray",imgGray)
    #to two-dimensional
    
    _,imgInv=cv2.threshold(imgGray,0,255,cv2.THRESH_BINARY_INV)
    
    #SYNTAX => cv2.threshold(src,threshold,maxval,type,dtype=None)
    #cv2.THRESH_BINARY px<threshold => 0 px px>threshold => 255 px
    #cv2.THRESH_BINARY_INV => just the opposite
    # px in the range of 0 goes 255 and vice versa
    #cv2.THRESH_TRUNC => threshold 200 means upto 200 same, greater than that changes to 200
    #cv2.THRESH_TOZERO => less than threshold to 0, greater than that remains same
    #cv3.THRESH_TOZERO_INV opposite
    #cv2.imshow("imgInv",imgInv)
    
    imgInv=cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    
    #the same grayscale value is used for all three color channels, resulting in a grayscale-looking BGR image.
    #127 px => (127,127,127) => same gray px except the shape differs
    #cv2.imshow("imgInvBGR",imgInv)
    #print(img.shape,imgInv.shape)
    
    img=cv2.bitwise_and(img,imgInv)
    
    
    #0, (1-255)=1 1(Has to be other than 0) && 1(imgInv) = 1
    #cv2.imshow("imgAND",img)
    
    img=cv2.bitwise_or(img,imgCanvas)
    
    # if any one px is 1, then 1 => img already has 1
    #cv2.imshow("imgOR",img)
    cv2.imshow("AIR WRITING",img)
    #cv2.imshow("Final imgInv",imgCanvas)
    
cap.release()
cv2.destroyAllWindows()

#import cv2
#img=cv2.imread("C:\\Users\\kishore prashanth\\Downloads\\ABC.png")
#img=cv2.resize(img,(128,128))
#cv2.imshow("Image",img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.imwrite("ROI.png",img)

