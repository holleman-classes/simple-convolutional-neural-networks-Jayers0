import tensorflow as tf
import numpy as np
import sys

# Load the model
model1 = tf.keras.models.load_model('best_model.h5')

# Function to preprocess the image
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(32, 32))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Class labels
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Function to perform classification
def classify_image(image_path):
    # Preprocess the image
    img_array = preprocess_image(image_path)
    
    # Perform classification
    predictions = model1.predict(img_array)
    
    # Get the predicted class label
    predicted_class = np.argmax(predictions)
    predicted_label = class_labels[predicted_class]
    
    return predicted_label, predictions[0]

# Path to the image file


# Classify the image
predicted_label, confidence = classify_image("test_image_car.png")

# Print the result
print("Predicted class:", predicted_label)
print("Confidence scores:", confidence[class_labels.index(predicted_label)])
