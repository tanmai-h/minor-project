# from imutils import paths
import os
import random
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
import numpy as np
import keras
from scipy import ndimage, misc

def preprocess_image(img):
    img_input = img.astype(np.uint8)
    # (channel_b, channel_g, channel_r) = cv2.split(img)

    result = ndimage.maximum_filter(img_input, size=5)
    ret3,result = cv2.threshold(result,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    result = cv2.bitwise_not(result)

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(11, 11))
    clahe_g = clahe.apply(img_input)

    image = np.zeros((331,331,3))

    image[:,:,0] = img_input
    image[:,:,1] = clahe_g
    image[:,:,2] = result

    image = image.astype(np.uint8)

    image = img_to_array(image)

    return image

def predictions_to_type(predictions):
    prediction_value=0
    for i in range(len(predictions)):
        if(predictions[i]==1):
            prediction_value = i
    
    dict_diseases={
        0: "Nothing Found",
        1: "Atelectasis",
        2: "Cardiomegaly",
        3: "Effusion",
        4: "Infiltration", 
        5: "Mass",
        6: "Nodule",
        7: "Pneumonia",
        8: "Pneumothorax",
        9: "Consolidation",
        10: "Edema",
        11: "Emphysema",
        12: "Fibrosis",
        13: "Pleural_Thickening",
        14: "Hernia"
    }
    return dict_diseases[prediction_value]
