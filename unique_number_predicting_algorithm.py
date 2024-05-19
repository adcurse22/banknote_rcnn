import cv2
import numpy as np
from tensorflow import keras
import easyocr

import matplotlib.pyplot as plt
import matplotlib.patches as patches

DATA_PATH = 'data/1k_2.jpg'
model = keras.models.load_model('banknote_rcnn_2.h5')

#--------------------------BBOX PREDICTING--------------------------

def bounding_box_transformer(y_pred):
  x1 = y_pred[2] / 2 + y_pred[0]
  y1 = y_pred[1]
  width1 = y_pred[2] / 2
  height1 = y_pred[3] / 2

  x2 = x1
  y2 = height1 / 3.2 + y1
  width2 = width1
  height2 = height1

  y_pred = [x2, y2, width2, height2]
  
  return y_pred

img = cv2.imread(DATA_PATH)
img_resized = np.array(cv2.resize(img, [256, 192]))
image_to_predict = np.expand_dims(img_resized, axis=0)

bbox_predicted = bounding_box_transformer(model.predict(image_to_predict).tolist()[0])

#Bbox transformation with original image ratio
width_ratio = img.shape[1] / 256
height_ratio = img.shape[0] / 192

bbox_predicted = [bbox_predicted[0] * width_ratio,
                  bbox_predicted[1] * height_ratio,
                  bbox_predicted[2] * width_ratio,
                  bbox_predicted[3] * height_ratio]

#Crop the region of interest
def crop_image(img, bbox_data):
  x = int(bbox_data[0])
  y = int(bbox_data[1])
  width = int(bbox_data[2])
  height = int(bbox_data[3])
  
  cropped_img = img[y:y+height, x:x+width]
  
  return cropped_img

roi = crop_image(img, bbox_predicted)

#Visualisation
plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
plt.show()

#FUNCTION FOR OCR REALIZATION
def text_recognize(img):
    reader = easyocr.Reader(["ru"])
    result = reader.readtext(img, detail=1, allowlist='1, 2, 3, 4, 5, 6, 7, 8, 9, 0, А, Б, В, Г, Д, Е, Ё, Ж, З, И, Й, К, Л, М, Н, О, П, Р, С, Т, У, Ф, Х, Ц, Ч, Ш, Щ, Ъ, Ы, Ь, Э, Ю, Я')

    return result

predicted_text = text_recognize(roi)
predicted_filtered_text = ''

for i in range(len(predicted_text)):
  
  if predicted_text[i][-1] >= 0.8:
    predicted_filtered_text += predicted_text[i][-2]
    
print(predicted_filtered_text)
