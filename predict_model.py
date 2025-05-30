import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model in .keras format
model = tf.keras.models.load_model('eye_disease_efficientnetb0_final.keras')

# Class names (must match training order)
class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

def predict_eye_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    print(f"Prediction: {predicted_class} (Confidence: {confidence:.2f}%)")

# Example usage
img_path = r"C:\Users\RJSSatheeshKumar\OneDrive - Sarda Metals & Alloys Ltd\Desktop\mediscan_eye_project\dataset\normal\2355_left.jpg"
predict_eye_disease(img_path)
