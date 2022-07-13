import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

import pandas as pd
import json
import cv2
import os
import numpy as np
import time
import json
# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf



model_path = "./inference/person.h5"

model = models.load_model(model_path, backbone_name='resnet50')

labels_to_names = {0:'Person' } 

color_coding = {'Person': (0, 255, 0)  }

img_path = "./person_test_image/"
#img_path = "C:/Users/Mayank/Downloads/Compressed/Amazon_Images/"
data = pd.DataFrame()
#mm = pd.read_csv("detect_Image_scale.csv")
#for img_name in mm["Image"]:
for img_name in os.listdir(img_path):
    # load image
    print(img_name)
    image = read_image_bgr(os.path.join(img_path,img_name))
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)
    
    # correct for image scale
    boxes /= scale
    labels_info = []
    # visualize detections
   # print(score)
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score <= 0.6:
            break

        b = box.astype(int)
    
        color = color_coding[labels_to_names[label]]
        caption = "{} {:.3f}".format(labels_to_names[label], score)

        labels_info.append( {"label": labels_to_names[label] ,"hex": color_coding[labels_to_names[label]],"width": int(b[2] - b[0]), "height": int(b[3] - b[1]) ,"x": int(b[0]), "y": int(b[1])})

        cv2.putText(draw, caption, (b[0],b[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), color, 10)
     
        data = data.append({'ImageID': img_name ,'XMin':b[0] ,'YMin':b[1],'XMax':b[2],'YMax':b[3] ,'LabelName':labels_to_names[label],'Conf': score }, ignore_index=True)     

    
    result = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./person_result/{}'.format(img_name), cv2.resize(result,(700,700)) )


#data.to_csv('detected_products_v1.csv', index=False)