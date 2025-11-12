import cv2
import os
import numpy as np
import random
from matplotlib import pyplot as plt
import uuid #for unique image names
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer,Conv2D,Dense,MaxPooling2D,Input,Flatten
import tensorflow as tf

#Avoid oom
#gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
 #   tf.config.experimental.set_memory_growth(gpu, True)

#print(gpus)

POST_PATH = os.path.join('data','positive')
NEG_PATH = os.path.join('data','negative')
ANC_PATH = os.path.join('data','anchor')

os.makedirs(POST_PATH,exist_ok=True)
os.makedirs(NEG_PATH,exist_ok=True)
os.makedirs(ANC_PATH,exist_ok=True)


for directory in os.listdir('lfw'):
    for file in os.listdir(os.path.join('lfw', directory)):
        EX_PATH = os.path.join('lfw',directory,file)
        NEW_PATH = os.path.join(NEG_PATH,file)
        os.replace(EX_PATH,NEW_PATH)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    #cut frame to 250x250
    frame = frame[170:420, 190:440, :]

    #anchor
    if cv2.waitKey(1) & 0xFF == ord('a'):
        imgname = os.path.join(ANC_PATH,'{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname,frame)

    #pozitive
    if cv2.waitKey(1) & 0xFF == ord('p'):
        imgname = os.path.join(POST_PATH, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)

    cv2.imshow('Image Collection',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

anchor = tf.data.Dataset.list_files(ANC_PATH+'\*jpg').take(300)
positive = tf.data.Dataset.list_files(POST_PATH+'\*jpg').take(300)
negative = tf.data.Dataset.list_files(NEG_PATH+'\*jpg').take(300)

def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img

img = preprocess('data\\anchor\\d0a47191-bfff-11f0-899a-d43b04945308.jpg')
#plt.imshow(img)
#plt.show()

#(anchor, positive) => 1,1,1,1,1
#(anchor,negative) => 0,0,0,0,0

positives = tf.data.Dataset.zip((anchor,positive,tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor,negative,tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

def preprocess_twin(input_img,validation_img,label):
    return preprocess(input_img),preprocess(validation_img),label

#dataloader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)
#samples = data.as_numpy_iterator()
#samp = samples.next()
#plt.imshow(samp[1])
#plt.show()
#plt.imshow(samp[0])
#plt.show()
#print(samp[2])

#training partition
train_data = data.take(round(len(data)*0.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)
#train_samples = train_data.as_numpy_iterator()
#train_samp = train_samples.next()
#print(len(train_samp[0]))

#testing partition
test_data = data.skip(round(len(data)*0.7))
test_data = test_data.take(round(len(data)*0.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)
#PART 4