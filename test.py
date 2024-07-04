import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import numpy as np

# Încarcă modelul antrenat
model = tf.keras.models.load_model('/home/drumea/Desktop/Hackathon/train/diases.keras')

# Încarcă denumirile claselor
with open('/home/drumea/Desktop/Hackathon/train/diases.txt') as f:
    class_labels = f.read().splitlines()

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def predict_image_class(img_path):
    preprocessed_img = preprocess_image(img_path)
    predictions = model.predict(preprocessed_img)
    return predictions

def predict_class_for_folder(folder_path):
    predictions = []
    # Iterează prin toate imaginile din folder
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            predictions.append(predict_image_class(img_path))
    
    # Agregă predicțiile
    if predictions:
        avg_predictions = np.mean(predictions, axis=0)
        predicted_class_index = np.argmax(avg_predictions)
        predicted_class = class_labels[predicted_class_index]
        return predicted_class
    else:
        return "Nicio imagine validă găsită în folder."

# Calea către folderul cu imagini
folder_path = '/home/drumea/Desktop/Hackathon/saved image'  # Actualizează cu calea către folderul tău cu imagini

# Realizează predicții pe baza imaginilor din folder
predicted_class = predict_class_for_folder(folder_path)

# Afișează rezultatele
print(f'Rezultate pentru imaginile din {folder_path}:')
print(f'Clasa prezisă: {predicted_class}')
