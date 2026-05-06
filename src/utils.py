from PIL import Image
import numpy as np

def preprocess_image(uploaded_file, size=(224, 224)):
    try:
        img = Image.open(uploaded_file).convert("RGB")

        img = img.resize(size)

        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        return img, img_array

    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")