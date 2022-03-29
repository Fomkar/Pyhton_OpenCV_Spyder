# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 13:14:37 2022

@author: Gitek_Micro
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 13:02:54 2022

@author: Gitek_Micro
"""

import numpy as np
import cv2

import os

# Görüntüleri okuma ve gösterme
currentDir = 'C:/Users/Gitek_Micro/Desktop/Fındık Veri Seti'
print(currentDir)
os.chdir(currentDir)
files = os.listdir()
idx =0 
for i,f in enumerate(files):
    if f.endswith(".bmp"):
        # The input image.
        print(f)
        image = cv2.imread(f, 1)
        if(idx==0):
            break

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

cv2.imshow("HSV", hsv)
cv2.waitKey(0)


cv2.destroyAllWindows()