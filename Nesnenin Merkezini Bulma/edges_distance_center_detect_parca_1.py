# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:57:45 2023

@author: Gitek_Micro
"""


import cv2 # opencv kütüphanesi
import numpy as np #matris ve dizi kütüphanesi
from datetime import datetime #zaman kütüphanesi
import time
import os
start_t = time.time()

start1 = datetime.now()
path = r"D:\otomotiv_numune\son"
os.chdir(path)
file = open("bloba_sayısı.txt", "a")
idx =0 
a = 0

circle_index = 1

square_index = 0
counter = 0
kernel = np.ones((5,5),np.uint8)
for filename in os.listdir(path):
    if filename.endswith("Image_1_i.bmp"):
        
       
        image = cv2.imread(filename,1)
        cv2.namedWindow("Original_image",cv2.WINDOW_NORMAL)
        cv2.imshow("Original_image", image)
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cv2.namedWindow("Gray_image",cv2.WINDOW_NORMAL)
        cv2.imshow("Gray_image", gray_image)
        # cv2.imshow("Original_image"+ str(a),image)
        
        gray_image = cv2.medianBlur(gray_image, 3)
        # gray_image = cv2.bilateralFilter(gray_image, 9,75,75)
        
        # edges = cv2.Canny(gray_image,15,15)
        # cv2.namedWindow("Canny image",cv2.WINDOW_NORMAL)
        # cv2.imshow("Canny image",edges)
        
        # erode = cv2.erode(edges, (5,5),iterations = 5)
        
        # cv2.namedWindow("Dilate_image",cv2.WINDOW_NORMAL)
        # cv2.imshow("Dilate_image",dilate)
        
        _,threshold = cv2.threshold(gray_image, 60, 255, cv2.THRESH_BINARY_INV)

        cv2.namedWindow("Threshold image",cv2.WINDOW_NORMAL)
        cv2.imshow("Threshold image",threshold)
        
        # median_image = cv2.medianBlur(threshold, 9)
        # image = cv2.bilateralFilter(image, 9,75,75)
        
        # edges = cv2.Canny(image,75,25)
        # cv2.namedWindow("Canny image",cv2.WINDOW_NORMAL)
        # cv2.imshow("Canny image",edges)
        

        dilate = cv2.dilate(threshold, (5,5),iterations = 13)
        cv2.namedWindow("Dilate_image",cv2.WINDOW_NORMAL)
        cv2.imshow("Dilate_image",dilate)


        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        contours, hierarchy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
#cnt = contours[0]
        if(len(contours) == 0):
            counter = counter + 1
        print(len(contours))
        i = 0
        contours_array =[]
        for cnt in contours:
            # print("blob var")
            area = cv2.contourArea(cnt)
            # print(contours[i])
            # print(area)
            
            if (area > 1000):
                # print(cnt)
                x,y,w,h = cv2.boundingRect(cnt)
                print("X0 : " ,x, "Y0 : ",y,"\nX1: ",x+w," Y1 : ",y+h)
                contours_array.append([x, y, w,h])
                i +=1
                y2 = int(y + h/2)

                
        centerx = int(((contours_array[circle_index][0]) + (contours_array[circle_index][2] / 2)))
        centery = int(((contours_array[circle_index][1])))
        radius = int(w/2)
        center_square_x = int(((contours_array[square_index][0]) + (contours_array[square_index][2] / 2)))
        y_square = contours_array[square_index][1]
        cv2.circle(image,(center_square_x,y_square) ,15 , (0, 255, 0), -1)
        
        cv2.circle(image,(centerx,centery) ,15 , (0, 0, 255), -1)
        # cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,250),5)
        
        cv2.putText(image, "Radius :" + str(radius), (x-25,y-20), cv2.FONT_HERSHEY_SIMPLEX , 1,  
         (0,215,255), 2, cv2.LINE_AA, False) 
        
        
        
        # cv2.line(image, (x, y), (x+w, y2), (4, 0, 225), 5)
        
        
        cv2.line(image, (centerx, centery), (center_square_x,y_square), (4, 255, 225), 5)
        uzunluk = centery - y_square
        cv2.putText(image, "{}".format(int(np.sqrt((centerx - center_square_x) ** 2 + (centery - y_square) ** 2))), (100, 100),
        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        # cv2.circle(image,(centerx,centery) ,radius , (255, 215, 0), 3)
        print("Radius :", w)
        print("merkez x :", centerx)
        print("merkez y :" ,y + h /2)
        cv2.namedWindow('Original image',cv2.WINDOW_NORMAL)
        cv2.imshow("Original image",image)
        cv2.imwrite("parca_1_i.jpg", image)
        cv2.waitKey(0)


cv2.waitKey(0)  
cv2.destroyAllWindows()


""" Çalışan kısım
               if x <= 311:
                   cv2.circle(image,(centerx,y) ,15 , (0, 0, 255), -1)
               else:
                   cv2.circle(image,(centerx,centery) ,15 , (0, 0, 255), -1)
"""
