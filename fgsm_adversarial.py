# import the necessary packages
from pyimagesearch.simplecnn import SimpleCNN
from pyimagesearch.simplecnn import ComplexCNN
from pyimagesearch.fgsm import generate_image_adversary
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from keras.models import load_model
import numpy as np
import cv2
import sys

# load MNIST dataset and scale the pixel values to the range [0, 1]
print("[INFO] loading MNIST dataset...")
(trainX, trainY), (testX, testY) = mnist.load_data()

trainX = trainX / 255.0
testX = testX / 255.0

trainX = trainX[:10000]
trainY = trainY[:10000]

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

model.save('10000.h5')

# model = load_model('10000.h5')
 
# make predictions on the testing set for the model trained on
# non-adversarial images
(loss, acc) = model.evaluate(x=testX, y=testY, verbose=0)
print("[INFO] loss: {:.4f}, acc: {:.4f}".format(loss, acc))

# loop over a sample of our testing images
epsilons = [0.15]
for eps in epsilons:
    for i in range(len(testX)):
        # grab the current image and label
        image = testX[i]
        label = testY[i]
        # generate an image adversary for the current image and make
        # a prediction on the adversary
        adversary = generate_image_adversary(model,
            image.reshape(1, 28, 28, 1), label, eps=eps)
        pred = model.predict(adversary)

        # scale both the original image and adversary to the range
        # [0, 255] and convert them to an unsigned 8-bit integers
        adversary = adversary.reshape((28, 28)) * 255
        adversary = np.clip(adversary, 0, 255).astype("uint8")

        imagePred = label.argmax()
        adversaryPred = pred[0].argmax()
        
        # if imagePred == adversaryPred:
        #     pass
        
        cv2.imwrite('eps_0.15/{}_{}_{}.jpg'.format(imagePred, adversaryPred,i), adversary)
        
        if i == 3001:
            break