from keras.models import load_model
from tensorflow.keras.utils import to_categorical
import glob
import cv2
import numpy as np
import re

model = load_model('10000.h5')
files = glob.glob("eps_0.15/*.jpg")

test_img = []
test_label = []

for file in files:
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    test_img.append(img)
    num = re.sub(r"\D", "", file)
    test_label.append(num[3])

test_img = np.array(test_img).astype('float32') / 255
test_img = test_img.reshape(len(test_img),28,28,1)
test_label = to_categorical(test_label)

test_loss, test_acc = model.evaluate(test_img, test_label, verbose=0)

print(test_acc)
print(test_loss)