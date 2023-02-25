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
drawSelectColor=(102, 0, 77) #white
brushSize=3
##################

def preprocess():
    image = cv2.imread('D:\\SRP\\HandTrackingProject\\ROI.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilate = cv2.dilate(thresh, kernel, iterations=6)
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        ROI = image[y:y+h, x:x+w]
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
        if fingers[0]==False and fingers[1]==False and fingers[2]==False and fingers[3]==False and fingers[4]==False:
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
    
    #cv2.imshow("imgCanvas",imgCanvas)
    imgGray=cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("imgGray",imgGray)
    _,imgInv=cv2.threshold(imgGray,0,255,cv2.THRESH_BINARY_INV)
    #cv2.imshow("imgInv",imgInv)
    imgInv=cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    #cv2.imshow("imgInvBGR",imgInv)
    #print(img.shape,imgInv.shape)
    img=cv2.bitwise_and(img,imgInv)
    #cv2.imshow("imgAND",img)
    img=cv2.bitwise_or(img,imgCanvas)
    #cv2.imshow("imgOR",img)
    cv2.imshow("AIR WRITING",img)
    #cv2.imshow("Final imgInv",imgInv)
    
cap.release()
cv2.destroyAllWindows()

