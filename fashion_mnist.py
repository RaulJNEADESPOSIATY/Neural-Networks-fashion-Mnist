#basic gpu with new loop

from google.colab import files
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from google.colab import drive
import cv2
from PIL import ImageOps
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam


device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


drive.mount('/content/drive')
My_images = ['/content/drive/MyDrive/tshirt.jpg', '/content/drive/MyDrive/shoes.jpg', '/content/drive/MyDrive/trousers_maroon.jpg']


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


class_names = ['Tshirt/top', "pullover", "bag", "shoes", 'trousers', 'dress', 'coat',
              'sandal', 'shirt', 'ankleboot']




train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)), # input
                        keras.layers.Dense(128, activation='relu'), #hidden
                        keras.layers.Dense(90, activation='relu'),
                        keras.layers.Dense(60, activation='relu'),
                        keras.layers.Dense(30, activation='relu'),
                        keras.layers.Dense(10, activation='softmax') # output
                        ])
model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

prediction = model.predict(test_images)
predicted_label = np.argmax(prediction[1])
print(class_names[predicted_label])
print(test_labels[1])

for i in range(10):  # Check the first 10 images
      predicted_label = np.argmax(prediction[i])
      actual_label = test_labels[i]
      print(f"Image {i}: Predicted: {class_names[predicted_label]}, Actual: {class_names[actual_label]}")
for i in range(10):
    plt.figure()
    plt.imshow(test_images[i])
    plt.colorbar()
    plt.grid(False)
    plt.show()
test_loss, test_acc = model.evaluate(test_images, test_labels)





for image_path in My_images:
  image = Image.open(image_path).convert('L')  # Opens image and converts to grayscale
  image = image.resize((28, 28))  # Resizes the image to 28x28 pixels
  grey_array = np.array(image)  # Converts the image to a NumPy array

  # Threshold the image to black and white for better recognition (optional)
  _, grey_array = cv2.threshold(grey_array, 127.0, 255.0, cv2.THRESH_BINARY)

  # Reshape and normalize the array for the model
  grey_array = grey_array.reshape(1, 28, 28) / 255.0

  plt.figure()
  plt.imshow(image)
  plt.colorbar()
  plt.grid(False)
  plt.show()

  prediction_tester = model.predict(grey_array)
  # class_index = np.argmax(prediction_tester)
  # class_name = class_names[class_index]
  # print(f"predicted class is: {class_name} ")

print("the testted result is: ")

model.fit(train_labels, prediction_tester, epochs=5)
prediction = model.predict(test_images)
predicted_label = np.argmax(prediction[1])
print(class_names[predicted_label])
print(test_labels[1])


