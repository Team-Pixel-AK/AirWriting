import streamlit as st
import cv2
import HandTrackingModule as htm
import time
import tensorflow as tf
import numpy as np
from PIL import Image
import csv
from streamlit_webrtc import webrtc_streamer,VideoTransformerBase


pTime=0 #previousTIme
cTime=0 #currTime
lmList=[] #List to store landmark id's & coordinates
xp,yp=0,0 #Initial points to draw
drawSelectColor=(102, 0, 102)
brushSize=3
imgCanvas=np.zeros((360,480,3),np.uint8)
cpy_imgCanvas=np.zeros((360,480,3),np.uint8)


def process(img):

    # img=cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
    detector=htm.handDetector(detectionCon=0.4)


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
                print("Prepare")
                
                
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
                #preprocess()

            #CLOSE mode:
            # if fingers[0]and fingers[1] and fingers[2] and fingers[3] and fingers[4]:
            #     cv2.putText(img,"CLOSE",(800,50),cv2.FONT_HERSHEY_PLAIN,2,(255, 255, 0),3)
            #     cap.release()
            #     cv2.destroyAllWindows()
                   


    


    return img



class VideoTransformer(VideoTransformerBase):

    
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = process(img)

        return img



webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

#webrtc_streamer(key="example")