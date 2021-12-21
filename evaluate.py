from keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import glob
import cv2
import numpy as np
import re

for i in range(3,7):
    model = load_model('models/model_{}.h5'.format(i))

    (trainX, trainY), (testX, testY) = mnist.load_data()
    testX = np.array(testX).astype('float32') / 255
    testX = testX.reshape(len(testX),28,28,1)
    testY = to_categorical(testY)
    test_loss, test_acc = model.evaluate(testX, testY, verbose=0)
    print(test_acc)
    print(test_loss)

#files = glob.glob("sample_2/*.jpg")
# test_img = []
# test_label = []

# for file in files:
#     img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
#     test_img.append(img)
#     num = re.sub(r"\D", "", file)
#     test_label.append(num[1])

# test_img = np.array(test_img).astype('float32') / 255
# test_img = test_img.reshape(len(test_img),28,28,1)
# test_label = to_categorical(test_label)
# test_loss, test_acc = model.evaluate(test_img, test_label, verbose=0)
