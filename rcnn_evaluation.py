import cv2
import numpy as np
from tensorflow import keras
import easyocr
import matplotlib.pyplot as plt

DIR_PATH = 'rcnn_evaluation_data/'
model = keras.models.load_model('banknote_rcnn_3.h5')

y_val = ['Ак2444841', 'БЧ3386314', 'ЧЧ0223533', 'ЧЬ4916890', 'ЧЭ4827452', 'ЭГ9508677', 'ЭИ0623432', 'ВЕ7663735', 'ЭО0286669', 'ЯЗ6158172', 'ЯЛ6847387', 'эх4511828', 'ЗЛ5024897', 'ис0434958', 'ал6103574', 'ЬС4992583', 'нс1017796', 'МТ5621855', 'ЧО6498426', 'ВТ8819783', 'кь1861322', 'аЬ7177257', 'ТЯ8044000', 
         'ЗЛ9139880', 'ЭО2257946', 'хи9749643', 'мб7662425', 'па7353957', 'мп2814276', 'ЗИ4628211', 'МА4941252', 'ГЛ6992604', 'ЭО4206147', 'ЧЯ0906061', 'ЗТ3962066', 'ЧО0280553', 'ЛК7748851', 'ХЯ9916052', 'ЭЗ1826047', 'эх0859703']

#--------------------------BBOX PREDICTING--------------------------

def bounding_box_transformer(y_pred):
  x1 = y_pred[2] / 2 + y_pred[0]
  y1 = y_pred[1]
  width1 = y_pred[2] / 2
  height1 = y_pred[3] / 2

  x2 = width1 / 3.5 + x1
  y2 = height1 / 4 + y1
  width2 = width1
  height2 = height1 / 2

  y_pred = [x2, y2, width2, height2]
  
  return y_pred

for filename in range(0, 40):
  img = cv2.imread(DIR_PATH + str(filename) + '.jpg')
  img_resized = np.array(cv2.resize(img, [170, 128]))
  
  image_to_predict = np.expand_dims(img_resized, axis=0)

  bbox_predicted = bounding_box_transformer(model.predict(image_to_predict).tolist()[0])

  #From relative coords to absolute
  (h, w) = img_resized.shape[:2]
  adjusted_bbox = [bbox_predicted[0] * w,#top_x
                  bbox_predicted[1] * h,#top_y
                  bbox_predicted[2] * w,#top_x
                  bbox_predicted[3] * h]#top_y

  #Bbox transformation with original image ratio
  width_ratio = img.shape[1] / 170
  height_ratio = img.shape[0] / 128

  #Scaling
  bbox_predicted = [adjusted_bbox[0] * width_ratio,
                    adjusted_bbox[1] * height_ratio,
                    adjusted_bbox[2] * width_ratio,
                    adjusted_bbox[3] * height_ratio]

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

#--------------------------EASYOCR--------------------------
  
  def text_recognize(img):
      reader = easyocr.Reader(["ru"])
      result = reader.readtext(img, detail=1)

      return result

  predicted_text = text_recognize(roi)
  predicted_filtered_text = ''
  print(predicted_text)

  #Set confidence threshold
  for (bbox, text, prob) in predicted_text:
    
    if prob >= 0.8:
      text = text.replace(' ', '')
      predicted_filtered_text += text
      
  print('STATUS', 'TRUE_CODE', 'PREDICTED_CODE', 'FILENAME')
  
  if predicted_filtered_text == y_val[filename]:
    print('ВЕРНО', y_val[filename], predicted_filtered_text, filename)
  
  else:
    print('НЕВЕРНО', y_val[filename], predicted_filtered_text, filename)