# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 12:53:01 2022

@author: Gitek_Micro
"""
#Kütüphaneler
import cv2
import numpy as np
import os
from datetime import datetime
import time

# Görüntüleri okuma ve gösterme
currentDir = 'C:/Users/Gitek_Micro/Desktop/Fındık Kesme'
print(currentDir)
os.chdir(currentDir)
files = os.listdir()
for i,f in enumerate(files):
    if f.endswith(".bmp"):
        print(f)
        image = cv2.imread(f)
        #cv2.imshow('Original', img)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        _,trehsold = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        cv2.imshow('Thresh image',trehsold)
        contours, hierarchy = cv2.findContours(trehsold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        #cnt = contours[0]
        idx =0 
        for cnt in contours:
              area = cv2.contourArea(cnt)
              print(area)
              if(area >5500):
                  idx += 1
                  x,y,w,h = cv2.boundingRect(cnt)
                  roi=image[y:y+h,x:x+w]
                  #cv2.imwrite(str(idx) + '.jpg', roi)
                  #cv2.rectangle(image,(x,y),(x+w,y+h),(0,25,255),5)
        #         cv2.imwrite('blob_fistik.jpg', image)
                  #cv2.imshow('img bounding',roi)
   
                 
                   

cv2.destroyAllWindows()