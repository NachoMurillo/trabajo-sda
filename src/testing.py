import sys, time
import numpy as np
import cv2
import config.params as params
from keras import layers
from keras import models
import h5py
from src.circledetection import circle_detection
import matplotlib.pyplot as plt

# Open Image for testing

img = cv2.imread("img/0.png",1)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


pred_act = []

def signal_type(prediction):
  # Vector con los nombres de las señales:
  signal = np.array(["Velocidad máxima 20 Km/h", "Velocidad máxima 30 Km/h", "Velocidad máxima 50 Km/h", "Velocidad máxima 60 Km/h", "Velocidad máxima 70 Km/h",
            "Velocidad máxima 80 Km/h", "Fin de limitación de velocidad máxima 80 Km/h", "Velocidad máxima 100 Km/h", "Velocidad máxima 120 Km/h", "Adelantamiento prohibido",
            "Adelantamiento prohibido para camiones", "Intersección con prioridad", "Calzada con prioridad", "Ceda el paso", "STOP", "Circulación prohibida en ambos sentidos",
            "Prohibición de acceso a vehículos destinados a transporte de mercancías", "Entrada prohibida", "Otros peligros", "Curva peligrosa hacia la izquierda",
            "Curva peligrosa hacia la derecha", "Curvas peligrosas hacia la izquierda", "Perfil irregular", "Pavimento deslizante", "Estrechamiento de calzada por la derecha",
            "Obras", "Proximidad de semáforo", "Peatones", "Niños", "Ciclistas", "Pavimento deslizante por hielo o nieve", "Paso de animales en libertad", "Fin de prohibiciones",
            "Sentido obligatorio derecha", "Sentido obligatorio izquierda", "Sentido obligatorio recto", "Recto y derecha únicas direcciones permitidas",
            "Recto e izquierda únicas direcciones permitidas", "Paso obligatorio derecha", "Paso obligatorio izquierda", "Intersección de sentido giratorio-obligatorio",
            "Fin de prohibición de adelantamiento", "Fin de prohibición de adelantamiento para camiones"])
  # Se asocia el número obtenido en la predicción con el nombre de la señal:
  if len(prediction) > 0:
    for k in range(0,len(prediction)):
      if prediction[k] < 10:
        print(str(prediction[k]) + "   ==>  " + str(signal[prediction[k]]))
      else:
        print(str(prediction[k]) + "  ==>  " + str(signal[prediction[k]]))
  else:
    print(str(prediction) + "  ==>  " + str(signal[prediction]))
  
def image_preproc(img, coef = None, width = None, height = None, inter = cv2.INTER_AREA):
    dim = (width,height)
    # RGB to Gray image conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # resize the image
    img_prep = cv2.resize(gray, dim, interpolation = inter)
    # rescale the image
    img_prep.astype('float32') # Convierte a float32
    img_prep = img_prep/coef # Escalado
    # return the resized image
    return img_prep

# cargar json y crear el modelo
# Nombre del archivo:
nombre = "config/red_neuronal" # PONER DIRECCIÓN EN DISCO DE RED NEURONAL ENTRE ""

json_file = open(nombre + ".json", 'r')
model_json = json_file.read()
json_file.close()
model = models.model_from_json(model_json)

# # cargar pesos al nuevo modelo
model.load_weights(nombre + ".h5")
print("Cargado modelo desde disco.")
    
ancho = 64
alto = 64
image = image_preproc(img, coef = 255, width = ancho, height = alto)
image = image.reshape([-1,ancho, alto,1])
predictions = model.predict(image, batch_size=1, verbose=0) # Obtiene los 43 porcentajes para la imagen
pred_max = np.argmax(predictions, axis=-1) # Se queda con la que tiene mayor porcentaje
# Imprime la clase predicha y la imagen original:
print("La señal predicha es de la clase: ")
pred_act.append(pred_max)
signal_type(pred_act)
