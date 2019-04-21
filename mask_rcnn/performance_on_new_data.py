
# coding: utf-8

# In[ ]:


import os 
import sys
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from imgaug import augmenters as iaa
from tqdm import tqdm
import pandas as pd 
import glob
from PIL import Image


# In[ ]:


DATA_DIR = '/kaggle/input/airbus-ship-detection'
#DATA_DIR = '/kaggle/input'
# Directory to save logs and trained model
ROOT_DIR = '/kaggle/working'


# In[ ]:


get_ipython().system('git clone https://www.github.com/matterport/Mask_RCNN.git')
os.chdir('Mask_RCNN')


# In[ ]:


# Import Mask RCNN
sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


# In[ ]:


# The following parameters have been selected to reduce running time for demonstration purposes 
# These are not optimal 

class DetectorConfig(Config):    
    # Give the configuration a recognizable name  
    NAME = 'airbus'
    
    GPU_COUNT = 1
    #IMAGES_PER_GPU = 9
    IMAGES_PER_GPU = 8
    
    #BACKBONE = 'resnet50'
    BACKBONE = 'resnet50'
    
    NUM_CLASSES = 2  # background and ship classes
    
    IMAGE_MIN_DIM = 384
    IMAGE_MAX_DIM = 384
    RPN_ANCHOR_SCALES = (8, 16, 32, 64)
    TRAIN_ROIS_PER_IMAGE = 64
    MAX_GT_INSTANCES = 14
    DETECTION_MAX_INSTANCES = 10
    DETECTION_MIN_CONFIDENCE = 0.95
    DETECTION_NMS_THRESHOLD = 0.0

    STEPS_PER_EPOCH = 15
    VALIDATION_STEPS = 10
    
    ## balance out losses
    LOSS_WEIGHTS = {
        "rpn_class_loss": 20.0,
        "rpn_bbox_loss": 0.8,
        "mrcnn_class_loss": 6.0,
        "mrcnn_bbox_loss": 1.0,
        "mrcnn_mask_loss": 1.2
    }

config = DetectorConfig()
config.display()


# In[ ]:


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from skimage.segmentation import mark_boundaries
from skimage.util import montage
from skimage.morphology import binary_opening, disk, label
import gc; gc.enable()


# In[ ]:


mytest_dir = '/kaggle/input/manual-test'
mytest_names = [f for f in os.listdir(mytest_dir)]


# In[ ]:


from keras.models import model_from_json,load_model
model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR)
class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode='inference', 
                          config=inference_config,
                          model_dir=ROOT_DIR)
model.load_weights('/kaggle/input/finalmodel/mask_rcnn_airbus_0022.h5',by_name=True)


# In[ ]:


# set color for class
def get_colors_for_class_ids(class_ids):
    colors = []
    for class_id in class_ids:
        if class_id == 1:
            colors.append((.941, .204, .204))
    return colors


# In[ ]:


test1 = imread('/kaggle/input/manual-test/test_image1.jpg')
#test1 = resize(test1,(384,384,3))
size1 = test1.shape[0]
_ = plt.imshow(test1)
size1


# In[ ]:


fig = plt.figure(figsize=(10, 100))

for i in range(20):

    image = imread(os.path.join('/kaggle/input/manual-test', mytest_names[i]))
    plt.subplot(20, 2, 2*i + 1)
    #visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
    #                            dataset.class_names,
    #                            colors=get_colors_for_class_ids(gt_class_id), ax=fig.axes[-1])
    plt.imshow(image)
    plt.subplot(20, 2, 2*i + 2)
    results = model.detect([image]) #, verbose=1)
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                'ship', r['scores'], 
                                colors=get_colors_for_class_ids(r['class_ids']), ax=fig.axes[-1])


# In[ ]:


for name in mytest_names:
    image = imread(os.path.join('/kaggle/input/manual-test', name))
    results = model.detect([image])
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                'ship', r['scores'], 
                                colors=get_colors_for_class_ids(r['class_ids']),ax=ax)
    plt.savefig(os.path.join(ROOT_DIR, name))
    plt.show()


# In[ ]:


get_ipython().system('rm -rf /kaggle/working/Mask_RCNN')

