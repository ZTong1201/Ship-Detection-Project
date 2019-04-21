import os
import gc
import numpy as np
import pandas as pd
import time

from PIL import image

from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.applications.resnet34 import ResNet34 as ResModel

from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Sequential, Model


TRAIN_PATH = '/Users/arc/train_v2/'
TEST_PATH = '/Users/arc/test_v2/'

TEST = os.listdir(TEST_PATH)
TRAIN = os.listdir(TRAIN_PATH)

epochs = 50
alpha  = 0.001
decay = alpha/epochs

sgd = optimizers.SGD(lr=alpha, momentum=0.9, decay=decay, nesterov=False)

train_segments = pd.read_csv('/Users/arc/train_ship_segmentations_v2.csv')

(train_segments.head())


train_segments['binary_flag'] = train_segments['EncodedPixels'].fillna(0)
train_segments.loc[train_segments['binary_flag']!=0,'binary_flag']=1


train_segments.head()


train = train.groupby('ImageId').sum().reset_index()
train.loc[train['binary_flag']>0,'binary_flag']=1





##TRAIN
data = np.empty((len(train_sample['ImageId']),256, 256,3), dtype=np.uint8)
ground = np.empty((len(train_sample['ImageId'])), dtype=np.uint8)

# print(len(data))
# print(len(ground))

DESIRED_TRAIN_IMAGES = [i for i in TRAIN if i in list(train_sample['ImageId'])]
for i,train_image in enumerate(DESIRED_TRAIN_IMAGES):

        im = Image.open(os.path.join(TRAIN_PATH,train_image))
        sized = im.resize((256,256)).convert('RGB')


        data[index]=size
        ground[index]=train_sample[train['ImageId'].str.contains(train_image)]['binary_flag'].iloc[0]


#print(data)


targets =ground.reshape(len(ground),-1)
enc = OneHotEncoder()
enc.fit(targets)
targets = enc.transform(targets).toarray()
print(targets.shape)


x_train, x_val, y_train, y_val = train_test_split(data,targets, test_size = 0.2)
x_train.shape, x_val.shape, y_train.shape, y_val.shape






######standard resnet loading
model = ResModel(weights = 'imagenet', include_top=False, input_shape = (256, 256, 3))



for layer in model.layers:
    layer.trainable = False

#### template for switching to 2 class softmax
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="tanh")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)

predictions = Dense(2, activation="softmax")(x)


model_ = Model(input = model.input, output = predictions)

###########

sgd = optimizers.SGD(lr=alpha, momentum=0.9, decay=decay, nesterov=False)
model_.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model_.summary()

model_.fit_generator(img_gen.flow(x_train, y_train, batch_size = 64),steps_per_epoch = len(x_train)/16, validation_data = (x_val,y_val), epochs = epochs )
model_.save('Resnet_binary_classification.h5')
