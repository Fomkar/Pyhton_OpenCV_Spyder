# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:18:03 2023

@author: Gitek_Micro
"""


import cv2 # opencv kütüphanesi
import numpy as np #matris ve dizi kütüphanesi
from datetime import datetime #zaman kütüphanesi
import time
import os
start_t = time.time()

start1 = datetime.now()
path = r"D:\otomotiv_numune"
os.chdir(path)
file = open("bloba_sayısı.txt", "a")
idx =0 
a = 0
counter = 0
kernel = np.ones((5,5),np.uint8)
for filename in os.listdir(path):
    if filename.endswith("2.bmp"):
        
       
        image = cv2.imread(filename,1)
        cv2.namedWindow("Original_image",cv2.WINDOW_NORMAL)
        cv2.imshow("Original_image", image)
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cv2.namedWindow("Gray_image",cv2.WINDOW_NORMAL)
        cv2.imshow("Gray_image", gray_image)
        # cv2.imshow("Original_image"+ str(a),image)
        
        image = cv2.medianBlur(image,3)
        # image = cv2.bilateralFilter(image, 1,75,75)
        
        edges = cv2.Canny(image,50,100)
        cv2.namedWindow("Canny image",cv2.WINDOW_NORMAL)
        cv2.imshow("Canny image",edges)
        
        # erode = cv2.erode(trehsold, (5,5),iterations = 15)
        dilate = cv2.dilate(edges, (5,5),iterations = 7)
        cv2.namedWindow("Dilate_image",cv2.WINDOW_NORMAL)
        cv2.imshow("Dilate_image",dilate)


        cv2.waitKey(0)
        # cv2.destroyAllWindows()

        contours, hierarchy = cv2.findContours(dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#cnt = contours[0]
        if(len(contours) == 0):
            counter = counter + 1
        print(len(contours))
        for cnt in contours:
            # print("blob var")
            area = cv2.contourArea(cnt)
            # print(area)
            
            if (area > 2500):
                x,y,w,h = cv2.boundingRect(cnt)
                y2 = int(y + h/2)
                centerx = int( x + w/2)
                centery = int( y + h/2)
                radius = int(w/2)
                cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,250),5)
                
                cv2.putText(image, "Radius :" + str(radius), (x-25,y-20), cv2.FONT_HERSHEY_SIMPLEX , 1,  
                 (0,215,255), 2, cv2.LINE_AA, False) 
                cv2.line(image, (x, y2), (x+w, y2), (4, 0, 225), 5)
                
                cv2.circle(image,(centerx,centery) ,radius , (255, 215, 0), 3)
                print("Radius :", w)
                print("merkez y :" ,y + h /2)
                cv2.namedWindow('Original image',cv2.WINDOW_NORMAL)
                cv2.imshow("Original image",image)
                cv2.waitKey(0)
#         cv2.imshow('img bounding'+str(idx),roi)
#         cv2.waitKey(0)


cv2.waitKey(0)  
cv2.destroyAllWindows()

