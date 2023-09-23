import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

path = "C:\\Users\\danhp\\PycharmProjects\\asimovAI\\trainingdata\\animalfaces\\train\\cat\\"
catImages =[]
catLabels = []

for filename in os.listdir(path):
    img = cv.imread(os.path.join(path, filename))
    if img is not None:
        img = img / 255.0  # Normalize between 0 and 1
        catImages.append(img)
        catLabels.append(1)

catImages = np.array(catImages)
catLabels = np.array(catLabels)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')  # 2 because we're assuming it's a binary classification (cats vs not-cats)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(catImages, catLabels, epochs=10, batch_size=32, validation_split=0.2)


# Load a new image
new_img = cv.imread('C:\\Users\\danhp\\PycharmProjects\\asimovAI\\trainingdata\\animalfaces\\train\\cat\\flickr_cat_000002.jpg')
new_img = cv.resize(new_img, (512, 512))
new_img = new_img / 255.0  # Normalize

# Predict
prediction = model.predict(np.expand_dims(new_img, axis=0))

if np.argmax(prediction) == 1:
    print("It's a cat!")
else:
    print("It's not a cat!")


# -----------------------------

# img = cv.imread('C:\\Users\\danhp\\PycharmProjects\\asimovAI\\trainingdata\\animalfaces\\train\\cat\\flickr_cat_000002.jpg')

# cv.imshow('Cat', img)

# cv.waitKey(0)

# script_dir = os.path.dirname(os.path.abspath('C:\\Users\\danhp\\PycharmProjects\\asimovAI\\trainingdata\\animalfaces\\train\\cat\\'))

# img = cv.imread('flickr_cat_000002.jpg')

# capture = cv.VideoCapture('0') reads your webcam

# capture = cv.videoCapture('video.mp4') reads a video file

# read video frame by frame

# while True:
#    isTrue, frame = capture.read()
#    cv.imshow('Video', frame)

#    if cv.waitKey(20) & 0xFF==ord('d'):
#        break

# capture.release()
# cv.destroyAllWindows()