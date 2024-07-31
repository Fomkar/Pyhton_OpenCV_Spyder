# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:36:36 2023

@author: Gitek_Micro
"""


import cv2 # opencv kütüphanesi
import numpy as np #matris ve dizi kütüphanesi
import os
from pypylon import pylon



idx =0 
a = 0
counter = 0
circle_index = 0
square_index = 0
kernel = np.ones((5,5),np.uint8)
# Pypylon get camera by serial number
serial_number = '40038474'
info = None
for i in pylon.TlFactory.GetInstance().EnumerateDevices():
    if i.GetSerialNumber() == serial_number:
        info = i
        break
else:
    print('Camera with {} serial number not found'.format(serial_number))

# VERY IMPORTANT STEP! To use Basler PyPylon OpenCV viewer you have to call .Open() method on you camera
if info is not None:
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(info))
    camera.Open()
    

    
# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
camera.AcquisitionFrameRateAbs.SetValue(True)
# camera.AcquisitionFrameRateEnable.SetValue = True
camera.AcquisitionFrameRateAbs.SetValue(15.0)
# camera.AcquisitionFrameRateAbs.SetValue = 5.0
# camera.Width.SetValue(720)
# camera.Height.SetValue = 540.0
# camera.Width.SetValue = 720.0
# camera.Width = 720
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned


# images = np.zeros((1000, 1080, 1440, 3), dtype=int)
# images = np.zeros((100, 540, 720, 3), dtype=int)

counter = 0
while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    
    
    
    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()
        # Get grabbed image
        # print("görüntü almaya başladı")
        img = img[0:1080,0:1429]

        # cv2.imwrite("image_4.jpg", img)

        gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        gray_image = cv2.medianBlur(gray_image, 7)


  
        _,threshold_circle = cv2.threshold(gray_image, 30, 250, cv2.THRESH_BINARY)

        cv2.namedWindow("Threshold image",cv2.WINDOW_NORMAL)
        cv2.imshow("Threshold image",threshold_circle)
    
    

        dilate_cirle = cv2.dilate(threshold_circle, (5,5),iterations = 1)
        dilate_cirle = cv2.erode(threshold_circle, (5,5),iterations = 1)
        # cv2.namedWindow("Dilate_image",cv2.WINDOW_NORMAL)
        # cv2.imshow("Dilate_image",dilate_cirle)


        # cv2.waitKey(0)
        

        contours_circle, hierarchy = cv2.findContours(dilate_cirle,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        

        if(len(contours_circle) == 0):
            counter = counter + 1
        i = 0
        contours_array_c =[]
        for cnt in contours_circle:
            area = cv2.contourArea(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            
            if (area > 7000 and x > 200 and y > 450 and x < 1100):
                # print(cnt)
                print(area)
                print("X0 : " ,x, "Y0 : ",y,"\nX1: ",x+w," Y1 : ",y+h)
                contours_array_c.append([x, y, w,h])
                i +=1
                y2 = int(y + h/2)
                
        centerx_circle = int(((contours_array_c[circle_index][0]) + (contours_array_c[circle_index][2] / 2)))
        centery_circle = int(((contours_array_c[circle_index][1])))
        radius = int(w/2) 
        cv2.circle(img,(centerx_circle,centery_circle) ,15 , (0, 255, 0), -1)
        
        # cv2.circle(image,(centerx,centery) ,15 , (0, 0, 255), -1)
        # cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,250),5)
        
        # cv2.putText(img, "Radius :" + str(radius), (x-25,y-20), cv2.FONT_HERSHEY_SIMPLEX , 1,  
        #   (0,215,255), 2, cv2.LINE_AA, False) 



        # cv2.line(image, (x, y), (x+w, y2), (4, 0, 225), 5)
        
        

        # cv2.circle(image,(centerx,centery) ,radius , (255, 215, 0), 3)
        print("Radius :", w)
        print("merkez x :", centerx_circle)
        print("merkez y :" ,y + h /2)
        # cv2.namedWindow('Original image',cv2.WINDOW_NORMAL)
        # cv2.imshow("Original image",img)
        # cv2.imwrite("parca_2.jpg", image)

        # cv2.imshow('img bounding'+str(idx),roi)
        # cv2.waitKey(0)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
                break

# cv2.destroyAllWindows()


        
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.namedWindow("Gray_image",cv2.WINDOW_NORMAL)
        cv2.imshow("Gray_image", gray_image)
        # cv2.imshow("Original_image"+ str(a),img)
        
        gray_image = cv2.medianBlur(gray_image, 7)

        _,threshold = cv2.threshold(gray_image, 45, 255,cv2.THRESH_BINARY_INV)
        
        # cv2.namedWindow("Threshold image",cv2.WINDOW_NORMAL)
        # cv2.imshow("Threshold image",threshold)
        

        

        dilate = cv2.dilate(threshold, (5,5),iterations = 2)
        dilate = cv2.erode(threshold, (5,5),iterations = 2)
        cv2.namedWindow("Dilate_image",cv2.WINDOW_NORMAL)
        cv2.imshow("Dilate_image",dilate)


        contours, hierarchy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        

        if(len(contours) == 0):
            counter = counter + 1
        print(len(contours))
        i = 0
        contours_array_s =[]
        for cnt in contours:
            # print("blob var")
            area = cv2.contourArea(cnt)
            # print(contours[i])
            # print(area)
            x,y,w,h = cv2.boundingRect(cnt)
            
            if (area > 10000 ):
                # print(cnt)
                
                print("X0 : " ,x, "Y0 : ",y,"\nX1: ",x+w," Y1 : ",y+h)
                contours_array_s.append([x, y, w,h])
                i +=1
                y2 = int(y + h/2)
                
        centerx_square = int(((contours_array_s[square_index][0]) + (contours_array_s[square_index][2] / 2)))
        centery_square = int(((contours_array_s[square_index][1])))
        radius = int(w/2) 
        cv2.circle(img,(centerx_square,centery_square) ,5 , (0, 0, 255), -1)
        
        # cv2.circle(image,(centerx_square,centery) ,15 , (0, 0, 255), -1)
        # cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,250),5)
        
        # cv2.putText(image, "Radius :" + str(radius), (x-25,y-20), cv2.FONT_HERSHEY_SIMPLEX , 1,  
        #  (0,215,255), 2, cv2.LINE_AA, False) 



        # cv2.line(img, (centerx_circle, centery_circle), (centerx_square, centery_square), (4, 0, 225), 5)
        pixsel = (int(np.sqrt((centerx_circle - centerx_square) ** 2 + (centery_circle - centery_square) ** 2)))
        uzunluk = int((int(np.sqrt((centerx_circle - centerx_square) ** 2 + (centery_circle - centery_square) ** 2))) / 35.5)
        cv2.putText(img, "uzunluk = {} mm".format(uzunluk), (100, 100),
        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
        

        cv2.putText(img, "pixel = {}".format(pixsel), (100, 160),
        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        # cv2.circle(image,(centerx,centery) ,radius , (255, 215, 0), 3)
        print("Radius :", w)
        print("merkez x :", centerx_square)
        print("merkez y :" ,y + h /2)
        cv2.namedWindow('Original image',cv2.WINDOW_NORMAL)
        cv2.imshow("Original image",img)
        # cv2.imwrite("parca_3.jpg", img)
        
        key = cv2.waitKey(1) & 0xFF
    #     # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            grabResult.Release()
            break
camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()
########################

