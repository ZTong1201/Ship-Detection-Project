
# coding: utf-8

# **<h2>Introduction**

# In this notebook, I refer to kaggle kernel: https://www.kaggle.com/meaninglesslives/airbus-ship-detection-data-visualizationtry to explore the Airbus Ship Detection Challenge data.

# In[183]:


import os
import sys
import random
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from tqdm import tqdm_notebook, tnrange
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from skimage.feature import canny
from skimage.filters import sobel,threshold_otsu, threshold_niblack,threshold_sauvola
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from scipy import signal

import cv2
from PIL import Image
import pdb
from tqdm import tqdm
import seaborn as sns
import os 
from glob import glob

import warnings
warnings.filterwarnings("ignore")


# <h2> Setting paths

# In[184]:


INPUT_PATH = '../input'
DATA_PATH = INPUT_PATH
TRAIN_DATA = os.path.join(DATA_PATH, "train_v2")
TRAIN_MASKS_DATA = os.path.join(DATA_PATH, "train_v2/masks")
TEST_DATA = os.path.join(DATA_PATH, "test_v2")
df = pd.read_csv(DATA_PATH+'/train_ship_segmentations_v2.csv')
path_train = '../input/train/'
path_test = '../input/test/'
train_ids = df.ImageId.values
df = df.set_index('ImageId')


# <h2> Some utility functions

# In[185]:


## Gets full path of a image given the image name and image type(test or train)
def get_filename(image_id, image_type):
    check_dir = False
    if "Train" == image_type:
        data_path = TRAIN_DATA
    elif "mask" in image_type:
        data_path = TRAIN_MASKS_DATA
    elif "Test" in image_type:
        data_path = TEST_DATA
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    if check_dir and not os.path.exists(data_path):
        os.makedirs(data_path)

    return os.path.join(data_path, "{}".format(image_id))

def get_image_data(image_id, image_type, **kwargs):
    img = _get_image_data_opencv(image_id, image_type, **kwargs)
    img = img.astype('uint8')
    return img

## Function to read image and return it 
def _get_image_data_opencv(image_id, image_type, **kwargs):
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname)
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

# https://github.com/ternaus/TernausNet/blob/master/Example.ipynb
def mask_overlay(image, mask):
    """
    Helper function to visualize mask
    """
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.75, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0    
    img[ind] = weighted_sum[ind]    
    return img


# <h2>**Plotting Images**

# Lets plot some random images from training set and then few more images with the mask overlayed on top of it.

# In[186]:



import traceback
import sys

## This function neatly displays the images in grid , we have option of showing masked / unmasked images.
def show_image(df,train_ids,show_masked = True , show_unmasked = True,plot_no_ship_images=False):
    ## We want to view 32 images in 4 rows
    nImg = 32  #no. of images that you want to display
    np.random.seed(42)
    if df.index.name == 'ImageId':
        df = df.reset_index()
    if df.index.name != 'ImageId':
        df = df.set_index('ImageId')

    _train_ids = list(train_ids)
    np.random.shuffle(_train_ids)
    tile_size = (256, 256)
    ## images per row
    n = 8
    alpha = 0.3

    ## Number of rows
    m = int(np.ceil(nImg * 1.0 / n))
    complete_image = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 3), dtype=np.uint8)
    complete_image_masked = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 3), dtype=np.uint8)

    counter = 0
    for i in range(m):
        ## For each row set up the row template for images
        ys = i*(tile_size[1] + 2)
        ye = ys + tile_size[1]
        j = 0
        while j < n:
            ## Now for each of images , load the image untill the we get 32 images
            counter += 1
            all_masks = np.zeros((768, 768))
            xs = j*(tile_size[0] + 2)
            xe = xs + tile_size[0]
            image_id = _train_ids[counter]
            ## For initial image exploration we would like to not have images with no ship , this can be toggle via the plot_no_ship_images option.
            if str(df.loc[image_id,'EncodedPixels'])==str(np.nan):
                if plot_no_ship_images:
                    j +=1
                else:    
                    continue
            else:
                j += 1
            img = get_image_data(image_id, 'Train')

            try:
                ## Depending on what type of images we want to see , compute the image matrix
                if show_unmasked:
                    img_resized = cv2.resize(img, dsize=tile_size)
                    img_with_text = cv2.putText(img_resized, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
                    complete_image[ys:ye, xs:xe, :] = img_with_text[:,:,:]
                    
                if show_masked:
                    img_masks = df.loc[image_id,'EncodedPixels'].tolist()
                    for mask in img_masks:
                        all_masks += rle_decode(mask)
                    all_masks = np.expand_dims(all_masks,axis=2)
                    all_masks = np.repeat(all_masks,3,axis=2).astype('uint8')*255

                    img_masked = mask_overlay(img, all_masks)        
                    img_masked = cv2.resize(img_masked, dsize=tile_size)

                    img_masked = cv2.putText(img_masked, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
                    complete_image_masked[ys:ye, xs:xe, :] = img_masked[:,:,:]

            except Exception as e:
                all_masks = rle_decode(df.loc[image_id,'EncodedPixels'])
                all_masks = np.expand_dims(all_masks,axis=2)*255
                all_masks = np.repeat(all_masks,3,axis=2).astype('uint8')

                img_masked = mask_overlay(img, all_masks)        

                img = cv2.resize(img, dsize=tile_size)
                img_masked = cv2.resize(img_masked, dsize=tile_size)

                img = cv2.putText(img, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
                complete_image[ys:ye, xs:xe, :] = img[:,:,:]

                img_masked = cv2.putText(img_masked, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
                complete_image_masked[ys:ye, xs:xe, :] = img_masked[:,:,:]
                
    ## Now plot images based on the options
    if show_unmasked:
        m = complete_image.shape[0] / (tile_size[0] + 2)
        k = 8
        n = int(np.ceil(m / k))
        for i in range(n):
            plt.figure(figsize=(20, 20))
            ys = i*(tile_size[0] + 2)*k
            ye = min((i+1)*(tile_size[0] + 2)*k, complete_image.shape[0])
            plt.imshow(complete_image[ys:ye,:,:],cmap='seismic')
            plt.title("Training dataset")
            
    if show_masked:
        m = complete_image.shape[0] / (tile_size[0] + 2)
        k = 8
        n = int(np.ceil(m / k))
        for i in range(n):
            plt.figure(figsize=(20, 20))
            ys = i*(tile_size[0] + 2)*k
            ye = min((i+1)*(tile_size[0] + 2)*k, complete_image.shape[0])
            plt.imshow(complete_image_masked[ys:ye,:,:])
            plt.title("Training dataset: Lighter Color depicts ship")

##Lets quickly test the function we just wrote            
show_image(df,train_ids)


# <h3>Plotting Ship Count

# In[187]:


df = df.reset_index()
df['ship_count'] = df.groupby('ImageId')['ImageId'].transform('count')
df.loc[df['EncodedPixels'].isnull().values,'ship_count'] = 0  #see infocusp's comment
sns.set_style("white")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.distplot(df['ship_count'],kde=False)
plt.title('Ship Count Distribution in Train Set')

print(df['ship_count'].describe())


# **Plotting Images: Based on Ship Count¶**

# **Lets begin with images with no ships¶**
# 

# In[188]:


images_with_noships = df[df["ship_count"] == 0].ImageId.values
show_image(df,images_with_noships,show_masked=False,plot_no_ship_images=True)


# **Lets begin with images with 1 to 5 ships¶**

# In[189]:


images_with_1_5 = df[df["ship_count"].between(1,5)].ImageId.values
show_image(df,images_with_1_5,show_unmasked=False,show_masked=True,plot_no_ship_images=True)


# **Training Set Images with Ship Count 5 to 10¶**

# In[190]:


images_with_5_10 = df[df["ship_count"].between(5,10)].ImageId.values
show_image(df,images_with_5_10,show_unmasked=False,show_masked=True,plot_no_ship_images=True)


# **Training Set Images with Ship Count greater than 10¶**

# In[191]:


images_with_greater_10 = df[df["ship_count"].between(10,16)].ImageId.values
show_image(df,images_with_greater_10,show_unmasked=False,show_masked=True,plot_no_ship_images=True)


# ## Summary

# In[192]:


df.head()


# In[193]:


print('Total records in the train_v2:',len(df))
print('Unique values in the train_v2:',len(df['ImageId'].unique()))


# In[196]:


# For total records in train
print('Ship vs no ship:',len(df[df['ship_count']!=0]),'vs',len(df[df['ship_count']==0]))


# In[197]:


not_empty = pd.notna(df.EncodedPixels)
print(not_empty.sum(), 'masks in', df[not_empty].ImageId.nunique(), 'images')
print((~not_empty).sum(), 'empty images in', df.ImageId.nunique(), 'total images')


# Next, I will preprocess the pictures using cv methods. Additionally, there are at least 65% photos without ship, so I decide to downsample the negative vlues without ship to balance the data set. 
# 

# In[199]:


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from skimage.segmentation import mark_boundaries
#from skimage.util import montage2d as montage
from skimage.morphology import binary_opening, disk, label
import gc; gc.enable() # memory is tight

montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
ship_dir = '../input'
train_image_dir = os.path.join(ship_dir, 'train')
test_image_dir = os.path.join(ship_dir, 'test')

def multi_rle_encode(img, **kwargs):
    '''
    Encode connected regions as separated masks
    '''
    labels = label(img)
    if img.ndim > 2:
        return [rle_encode(np.sum(labels==k, axis=2), **kwargs) for k in np.unique(labels[labels>0])]
    else:
        return [rle_encode(labels==k, **kwargs) for k in np.unique(labels[labels>0])]
# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_max_threshold:
        return '' ## no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return '' ## ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction
def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks

def masks_as_color(in_mask_list):
    # Take the individual ship masks and create a color mask array for each ships
    all_masks = np.zeros((768, 768), dtype = np.float)
    scale = lambda x: (len(in_mask_list)+x+1) / (len(in_mask_list)*2) ## scale the heatmap image to shift 
    for i,mask in enumerate(in_mask_list):
        if isinstance(mask, str):
            all_masks[:,:] += scale(i) * rle_decode(mask)
    return all_masks


# In[201]:


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (16, 5))
rle_0 = masks.query('ImageId=="00021ddc3.jpg"')['EncodedPixels']
img_0 = masks_as_image(rle_0)
ax1.imshow(img_0)
ax1.set_title('Mask as image')
rle_1 = multi_rle_encode(img_0)
img_1 = masks_as_image(rle_1)
ax2.imshow(img_1)
ax2.set_title('Re-encoded')
img_c = masks_as_color(rle_0)
ax3.imshow(img_c)
ax3.set_title('Masks in colors')
img_c = masks_as_color(rle_1)
ax4.imshow(img_c)
ax4.set_title('Re-encoded in colors')
print('Check Decoding->Encoding',
      'RLE_0:', len(rle_0), '->',
      'RLE_1:', len(rle_1))
print(np.sum(img_0 - img_1), 'error')

