# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 10:23:06 2022

@author: Gitek_Micro
"""

#Kütüphaneler


import numpy as np
import cv2

import os
# Görüntüleri okuma ve gösterme
currentDir = 'C:/Users/Gitek_Micro/Desktop/findik_icleri'
print(currentDir)
os.chdir(currentDir)
files = os.listdir()
idx =0 
for i,f in enumerate(files):
    if f.endswith(".bmp"):
        # The input image.
        print(f)
        # x = f.split("_")
        # # print(x[7])
        # y  = x[7].split(".")
        # print(y)
        # # print(y[0])
        # a = y[0]
        # print(a)
        image = cv2.imread(f, 1)
        gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        params = cv2.SimpleBlobDetector_Params()

        #Define thresholds
        #Can define thresholdStep. See documentation. 
        params.minThreshold = 60
        params.maxThreshold = 255

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 2000
        params.maxArea = 100000

        # Filter by Color (black=0)
        params.filterByColor = False  #Set true for cast_iron as we'll be detecting black regions
        params.blobColor = 0

        # Filter by Circularity


        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.5
        params.maxConvexity = 1

        # Filter by InertiaRatio
        params.filterByInertia = True
        params.minInertiaRatio = 0
        params.maxInertiaRatio = 1



        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray_image)
        img_with_blobs = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        
        
        counter=0
        print( "Bulunan Blob Sayisi : {}".format(int(len(keypoints))))
        #cv2.namedWindow("Keypoints", cv2.WINDOW_NORMAL)   
        #♣cv2.imshow("Keypoints", img_with_blobs)
        cv2.waitKey(250)
        if(len(keypoints)==0):
            os.remove(f)
            pass
        else:
            pass
    



cv2.waitKey(0)
cv2.destroyAllWindows()