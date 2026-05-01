import tensorflow as tf

def load_models():
    mobilenet = tf.keras.models.load_model("./models/models/mobilenet_pneumonia.keras")
    resnet = tf.keras.models.load_model("./models/models/resnet50_pneumonia.keras")
    return mobilenet, resnet

def predict(model, img_array):
    prob = model.predict(img_array)[0][0]
    label = "PNEUMONIA" if prob > 0.5 else "NORMAL"
    confidence = prob if prob > 0.5 else 1 - prob
    return label, confidence, prob