# import the necessary packages
from pyimagesearch.simplecnn import SimpleCNN
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
for i in range(trainX.shape[0]):
    th_train = trainX[i]
    th_train[trainX[i]<100] = 0
    th_train[trainX[i]>=100] = 255
    trainX[i] = th_train
for i in range(testX.shape[0]):
    th_train = testX[i]
    th_train[testX[i]<100] = 0
    th_train[testX[i]>=100] = 255
    testX[i] = th_train
trainX = trainX / 255.0
testX = testX / 255.0
# add a channel dimension to the images
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
# one-hot encode our labels
trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)

model = load_model('30000.h5')

# loop over a sample of our testing images
for i in np.random.choice(np.arange(0, len(testX)), size=(10,)):
    # grab the current image and label
    image = testX[i]
    label = testY[i]
    # generate an image adversary for the current image and make
    # a prediction on the adversary
    adversary = generate_image_adversary(model,
        image.reshape(1, 28, 28, 1), label, eps=0.1)
    pred = model.predict(adversary)

    # scale both the original image and adversary to the range
    # [0, 255] and convert them to an unsigned 8-bit integers
    adversary = adversary.reshape((28, 28)) * 255
    adversary = np.clip(adversary, 0, 255).astype("uint8")
    image = image.reshape((28, 28)) * 255
    image = image.astype("uint8")
    # convert the image and adversarial image from grayscale to three
    # channel (so we can draw on them)
    image = np.dstack([image] * 3)
    adversary = np.dstack([adversary] * 3)
    # resize the images so we can better visualize them
    image = cv2.resize(image, (96, 96))
    adversary = cv2.resize(adversary, (96, 96))
    
    # determine the predicted label for both the original image and
    # adversarial image
    imagePred = label.argmax()
    adversaryPred = pred[0].argmax()
    color = (0, 255, 0)
    # if the image prediction does not match the adversarial
    # prediction then update the color
    if imagePred != adversaryPred:
        color = (0, 0, 255)
    # draw the predictions on the respective output images
    cv2.putText(image, str(imagePred), (2, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 2)
    cv2.putText(adversary, str(adversaryPred), (2, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)
    # stack the two images horizontally and then show the original
    # image and adversarial image
    output = np.hstack([image, adversary])
    cv2.imshow("FGSM Adversarial Images", output)
    cv2.waitKey(0)