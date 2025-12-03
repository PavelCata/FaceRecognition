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
    key = cv2.waitKey(1) & 0xFF
    #anchor
    if key == ord('a'):
        imgname = os.path.join(ANC_PATH,'{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname,frame)

    #pozitive
    if key == ord('p'):
        imgname = os.path.join(POST_PATH, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)

    #quit
    cv2.imshow('Image Collection',frame)
    if key == ord('q'):
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

img = preprocess('data\\anchor\\84fea2e5-d05e-11f0-a710-d43b04945308.jpg')
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

# samples = data.as_numpy_iterator()
# samp = samples.next()
# #create 2 subplots to see anchor and positive/negative
# fig, ax = plt.subplots(1, 2, figsize=(6,3))
# ax[0].imshow(samp[0])
# ax[0].set_title("Anchor")
# ax[0].axis('off')
# ax[1].imshow(samp[1])
# ax[1].set_title("Positive/Negative")
# ax[1].axis('off')
# plt.show()

# #print label 0 - negative sample, 1 - positive sample
# print(samp[2])


#training partition
size = tf.data.experimental.cardinality(data).numpy()
train_data = data.take(round(size*0.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)
#train_samples = train_data.as_numpy_iterator()
#train_samp = train_samples.next()
#print(len(train_samp[0]))

#testing partition
test_data = data.skip(round(size*0.7))
test_data = test_data.take(round(size*0.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)


#Build Embedding Model
#Convolution -> MaxPooling -> Conv -> MaxPooling -> Conv -> MaxPooling -> Conv -> Flatten -> Dense
#Pipeline for feature extraction
def make_embedding():
    inp = Input(shape = (100,100,3), name  = 'inp_image')
    #intern while
    #learns simple details-> edges, lines, corners
    conv1 = Conv2D(64,(10,10),activation='relu')(inp)
    mp1 = MaxPooling2D(64,(2,2), padding = 'same')(conv1)
    
    #learns facial features -> curbs of the face, textures, shapes
    conv2 = Conv2D(128,(7,7),activation='relu')(mp1)
    mp2 = MaxPooling2D(64,(2,2), padding = 'same')(conv2)
    
    #complex features -> the shape of nose, eyes, lips
    conv3 = Conv2D(128,(4,4),activation='relu')(mp2)
    mp3 = MaxPooling2D(64,(2,2), padding = 'same')(conv3)
    
    #fine details -> distance between eyes, nose tip to chin
    conv4 = Conv2D(256,(4,4),activation='relu')(mp3)
    
    flat = Flatten()(conv4)
    dense = Dense(4096,activation='sigmoid')(flat)
    # Numeric code of the image
    #the model has an input image and outputs a vector of 4096 numbers
    return Model(inputs=inp,outputs=dense, name='embedding_model')

#print(make_embedding().summary())


#Distance Layer for the siamese network
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    #similarity calculation
    def call(self, input_embedding, validation_embedding): #input_embedding - anchor, validation_embedding - positive/negative
        return tf.math.abs(input_embedding - validation_embedding)
    

embedding = make_embedding()
def make_siamese_model():
    input = Input(name='input_img', shape=(100,100,3))
    validation = Input(name='validation_img', shape=(100,100,3))
    
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input), embedding(validation))
    
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input, validation], outputs=classifier, name='SiameseNetwork')

siamese_model = make_siamese_model()
#print(siamese_model.summary())

binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

@tf.function
def train_step(batch):
    #record all operations
    with tf.GradientTape() as tape:
        #get anchor and positive/negative image
        X = batch[:2]
        #get label
        y = batch[2]
        #forward pass
        yhat = siamese_model(X, training=True)
        #calculate loss
        loss = binary_cross_loss(y, yhat)
    pass 
    print(loss)

    grad = tape.gradient(loss, siamese_model.trainable_variables)
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    return loss

def train(data, EPOCHS):
    #loop over epochs
    #Epochs is the number of times the model sees the data
    for epoch in range(1,EPOCHS+1):
        print("\n Epoch {}/{}".format(epoch,EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))

        #loop over each batch
        for idx, batch in enumerate(data):
            loss = train_step(batch)
            progbar.update(idx+1, [("loss", loss)])

        #checkpoints
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


EPOCHS = 50
train(train_data, EPOCHS)