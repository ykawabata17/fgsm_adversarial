from pyimagesearch.simplecnn import ComplexCNN
from pyimagesearch.fgsm import generate_image_adversary
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from keras.models import load_model
import numpy as np
import glob
import re
import cv2

def model_create(img_file, origin, noise):
    # 元画像の読み込み
    (trainX, trainY), (testX, testY) = mnist.load_data()
    trainX = trainX / 255.0
    testX = testX / 255
    trainX = np.expand_dims(trainX, axis=-1)
    testX = np.expand_dims(testX, axis=-1)
    testY = to_categorical(testY,10)
    
    # ノイズ画像の読み込み
    trainX_noise = []
    trainY_noise = []
    imgs = glob.glob("{}/*.jpg".format(img_file))
    for img in imgs:
        num = re.sub(r"\D", "", img)
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        trainX_noise.append(img)
        trainY_noise.append(num[1])
    trainX_noise = np.array(trainX_noise).astype('float32') / 255
    trainX_noise = trainX_noise.reshape(len(trainX_noise), 28, 28, 1)
    trainY_noise = np.array(trainY_noise)
    
    trainX_org = trainX
    trainY_org = trainY
    testX = testX
    testY = testY
    trainX_noise = trainX_noise
    trainY_noise = trainY_noise
    
    trainX = np.append(trainX_org[origin:10000+origin], trainX_noise[:noise])
    trainY = np.append(trainY_org[origin:10000+origin], trainY_noise[:noise])
    trainX = trainX.reshape(15000, 28, 28, 1)
    
    random = np.arange(len(trainX))
    np.random.shuffle(random)
    trainX = trainX[random]
    trainY = trainY[random]
    trainY = to_categorical(trainY)

    opt = Adam(lr=1e-3)
    model = ComplexCNN.build()
    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])
    model.fit(trainX, trainY,
        validation_data=(testX, testY),
        batch_size=64,
        epochs=10,
        verbose=1)
    
    print(len(trainX))
    model.save('models/org{}_noise{}.h5'.format(origin, noise))

def main():
    model_create(img_file="sample_1", origin=30000, noise=5000)
    
if __name__=="__main__":
    main()
        