# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 13:25:35 2022

@author: Gitek_Micro
"""

#fındıkaları +20 pixel ile kes
import cv2
import numpy as np
import os

# Görüntüleri okuma ve gösterme
currentDir = 'C:/Users/Gitek_Micro/Desktop/Fındık_ici Veri Seti'
print(currentDir)
os.chdir(currentDir)
files = os.listdir()
idx =0 
for i,f in enumerate(files):
    if f.endswith(".bmp"):
        #print(f)
        image = cv2.imread(f)
        
        # cv2.imshow("Original image",image)
        # cv2.waitKey(0)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        
        _,trehsold = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        cv2.dilate(trehsold, (7,7),iterations = 15)
        # cv2.imshow("Treshold image", trehsold)
        # cv2.waitKey(0)
        contours, hierarchy = cv2.findContours(trehsold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        #cnt = contours[0]
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            #print(area)
            x,y,w,h = cv2.boundingRect(cnt)
            if(area >5500 and y-20 > 0 and y + h + 20 < 1080 and x-20 > 0 and x + w + 20 <1440):
                idx += 1
                
                roi=image[y - 20:y+ h +20,x-20:x+w+20]
                cv2.imwrite("C:/Users/Gitek_Micro/Desktop/findik_icleri/findik_" + str(idx) + '.bmp', roi)
                #cv2.rectangle(image,(x,y),(x+w,y+h),(0,25,255),5)
                #cv2.imshow('img bounding',roi)

cv2.destroyAllWindows()
