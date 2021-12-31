# import the necessary packages
from pyimagesearch.simplecnn import ComplexCNN
from pyimagesearch.fgsm import generate_image_adversary
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from keras.models import load_model
import numpy as np
import cv2

# load MNIST dataset and scale the pixel values to the range [0, 1]
print("[INFO] loading MNIST dataset...")
(trainX, trainY), (testX, testY) = mnist.load_data()

trainX = trainX / 255.0
testX = testX / 255.0

trainX = trainX[30000:40001]
trainY = trainY[30000:40001]

# add a channel dimension to the images
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
# one-hot encode our labels
trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)

# initialize our optimizer and model
print("[INFO] compiling model...")
opt = Adam(lr=1e-3)
model = ComplexCNN.build()
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])
# train the simple CNN on MNIST
print("[INFO] training network...")
model.fit(trainX, trainY,
    validation_data=(testX, testY),
    batch_size=64,
    epochs=10,
    verbose=1)

model.save('30000_40000.h5')