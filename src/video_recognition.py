# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 17:13:22 2021

@author: EQUIPO
"""

import sys, time
import numpy as np
import cv2
import numpy as np
import config.params as params
from keras.models import load_model
import h5py
from src.circledetection import circle_detection

def classify(img, model):
    pred = model.predict_classes([image])[0]
    #sign = classes[pred+1]
    sign = pred
    if sign:
        print(sign)
        
def grayscale(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  return img

def equalize(img):
  img = cv2.equalizeHist(img)
  return img        

def preprocessing(img):
  img = np.asarray(img)
  img = cv2.resize(img, (32, 32))
  img = grayscale(img)
  img = equalize(img)
  #normalize the images, i.e. convert the pixel values to fit btwn 0 and 1
  img = img/255
  return img        

if __name__ == "__main__":
    
    frame_rate = 30
    prev = 0
    captura = cv2.VideoCapture(0)
    ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    salida = cv2.VideoWriter('img/videoSalida.mp4',cv2.VideoWriter_fourcc(*'XVID'),15.0,(ancho,alto))
    
    model = load_model('config/my_model.h5')
    
    while (captura.isOpened()):
            time_elapsed = time.time() - prev
            ret, image = captura.read()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if time_elapsed > 1./frame_rate:
                prev = time.time()
                if ret == True:
                       cimg, det, cropped_image = circle_detection(image)
                       if det == 1:
                           cv2.imshow('video', cropped_image)
                           salida.write(cimg)
                           image = preprocessing(image)
                           image = image.reshape(1, 32, 32, 1)
                           #sclassify(image, model)
                       else:
                           cv2.imshow('video', cimg)
                           salida.write(cimg)
                       if cv2.waitKey(1) & 0xFF == ord('s'):
                            break
                else:break
                        
    captura.release()
    cv2.destroyAllWindows()