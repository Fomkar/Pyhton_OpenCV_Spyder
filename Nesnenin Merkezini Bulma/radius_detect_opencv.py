# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:28:42 2023

@author: Gitek_Micro
"""

import sys
import cv2 as cv
import numpy as np
def main(argv):
 
 default_file = 'image_1.bmp'
 filename = argv[0] if len(argv) > 0 else default_file
 # Loads an image
 src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
 # Check if image is loaded fine
 if src is None:
     print ('Error opening image!')
     print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
     return -1
     
 
 gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
 
 
 gray = cv.medianBlur(gray, 5)
 
 
 rows = gray.shape[0]
 circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
 param1=150, param2=150,
)


 a = 0
 if circles is not None:
     circles = np.uint16(np.around(circles))
     for i in circles[a, :]:
         center = (i[0], i[1])
         radius = i[2]
         print("x :", i[0])
         print("y :", i[1])
         print("RAdius ." ,radius)
         x1 = int(i[0] - i[2])
         y1 = int(i[1])
         x2 = int(i[0] + i[2])
         cv.line(src, (x2, y1), (x1, y1), (204, 255, 255), 5)
         print ("----------------------------------------")
         # circle center
         cv.circle(src, center, 1, (0, 100, 100), 3)
         # circle outline
         
         cv.circle(src, center, radius, (255, 0, 255), 3)
         a = 1
 cv.namedWindow("detected circles",cv.WINDOW_NORMAL)
 cv.imshow("detected circles", src)
 cv.waitKey(0)
 cv.destroyAllWindows()
 return 0
if __name__ == "__main__":
 main(sys.argv[1:])