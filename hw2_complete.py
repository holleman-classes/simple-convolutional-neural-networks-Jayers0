import tensorflow as tf
from tensorflow.keras.datasets import cifar10

import model1
import model2
import model3
import model50k

## 

def build_model1():
  # Add code to define model 1 using the function from model1.py
  model = model1.build_model1()
  return model

def build_model2():
  # Add code to define model 2 using the function from model2.py
  model = model2.build_model2()
  return model

def build_model3():
  # Add code to define model 3 using the function from model3.py
  model = model3.build_model3()
  ## This one should use the functional API so you can create the residual connections
  return model

def build_model50k():
  # Add code to define model with less than 50k parameters using the function from model50k.py
  model = model50k.build_model50k()
  return model

# no training or dataset construction should happen above this line
model50k = build_model50k()

if __name__ == '__main__':

  ########################################
  ## Add code here to Load the CIFAR10 data set
  # Include code to load the CIFAR-10 dataset here

  (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    
  # Normalize pixel values to range [0, 1]
  train_images, test_images = train_images / 255.0, test_images / 255.0

  ########################################
  ep = 50
  ## Build and train model 1
  model1 = build_model1()
  # compile and train model 1.
  model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  model1.fit(train_images, train_labels, epochs=ep, validation_data=(test_images, test_labels))
  print("Model 1 training completed.")

  ## Build, compile, and train model 2 (DS Convolutions)
  model2 = build_model2()
  model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  model2.fit(train_images, train_labels, epochs=ep, validation_data=(test_images, test_labels))
  print("Model 2 training completed.")
  
  ## Build, compile, and train model 3
  model3 = build_model3()
  model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  model3.fit(train_images, train_labels, epochs=ep, validation_data=(test_images, test_labels))
  print("Model 3 training completed.")

  
  model50k.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  model50k.fit(train_images, train_labels, epochs=ep, validation_data=(test_images, test_labels))
  model50k.save("best_model.h5")
  print("Model 50k training completed.")
