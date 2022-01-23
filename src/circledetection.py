# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 18:08:42 2021

@author: EQUIPO
"""

import numpy as np
import cv2 

def circle_detection(img):
    img = cv2.medianBlur(img,5)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT_ALT,1.2, 100,
                                param1=200,param2=0.95, minRadius=60)  
    if circles is not None:
        circles = np.uint16(np.around(circles))
        det = 1
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
            # crop image
            # TODO:  Arreglar extremos
            cropped_image = img[i[1]-i[2]-10:i[1]+i[2]+10,
                                i[0]-i[2]-10:i[0]+i[2]+10]
    else:
        det = 0 
        cropped_image = None
        
    return img, det, cropped_image

if __name__ == "__main__":
    img = cv2.imread("img/circulos.png")
    cimg, det = circle_detection(img)
    cv2.imshow("Imagen circulos", cimg)
    cv2.waitKey(0)