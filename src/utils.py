
import cv2
import numpy as np

def preprocess_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    img = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_array = img_rgb / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_rgb, img_array