
# coding: utf-8

# In[ ]:


from fastai.conv_learner import *
from fastai.dataset import *

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import auc,roc_curve
from sklearn.metrics import precision_recall_curve
import seaborn as sns


# In[ ]:


PATH = './'
TRAIN = '../input/airbus-ship-detection/train_v2/'
TEST = '../input/airbus-ship-detection/test_v2/'
SEGMENTATION = '../input/airbus-ship-detection/train_ship_segmentations_v2.csv'
exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg', 
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg']


# In[ ]:


nw = 4   #number of workers for data loader
arch = resnet34 #specify target architecture


# In[ ]:


train_names = [f for f in os.listdir(TRAIN) if f not in exclude_list]
test_names = [f for f in os.listdir(TEST) if f not in exclude_list]
#5% of data in the validation set is sufficient for model evaluation
tr_n, val_n = train_test_split(train_names, test_size=0.05, random_state=42)


# In[ ]:


class pdFilesDataset(FilesDataset):
    def __init__(self, fnames, path, transform):
        self.segmentation_df = pd.read_csv(SEGMENTATION).set_index('ImageId')
        super().__init__(fnames, transform, path)
    
    def get_x(self, i):
        img = open_image(os.path.join(self.path, self.fnames[i]))
        if self.sz == 768: return img 
        else: return cv2.resize(img, (self.sz, self.sz))
    
    def get_y(self, i):
        if(self.path == TEST): return 0
        masks = self.segmentation_df.loc[self.fnames[i]]['EncodedPixels']
        if(type(masks) == float): return 0 #NAN - no ship 
        else: return 1
    
    def get_c(self): return 2 #number of classes


# In[ ]:


def get_data(sz,bs):
    #data augmentation
    aug_tfms = [RandomRotate(20, tfm_y=TfmType.NO),
                RandomDihedral(tfm_y=TfmType.NO),
                RandomLighting(0.05, 0.05, tfm_y=TfmType.NO)]
    tfms = tfms_from_model(arch, sz, crop_type=CropType.NO, tfm_y=TfmType.NO, 
                aug_tfms=aug_tfms)
    ds = ImageData.get_ds(pdFilesDataset, (tr_n[:-(len(tr_n)%bs)],TRAIN), 
                (val_n,TRAIN), tfms, test=(test_names,TEST))
    md = ImageData(PATH, ds, bs, num_workers=nw, classes=None)
    #md.is_multi = False
    return md


# In[ ]:


sz = 384 #image size
bs = 32  #batch size

md = get_data(sz,bs)
learn = ConvLearner.pretrained(arch, md, ps=0.5) #dropout 50%
learn.opt_fn = optim.Adam
learn.unfreeze()
lr=np.array([1e-4,5e-4,2e-3])


# In[ ]:


state = torch.load('../input/resnet34384/resnet34_384.pth')
learn.model.load_state_dict(state)


# ### Validation Accuracy

# In[ ]:


log_val_probs,y = learn.predict_with_targs()
val_probs = np.exp(log_val_probs)[:,1]
val_pred = (val_probs > 0.5).astype(int)


# In[ ]:


def print_confusion_matrix(confusion_matrix, class_names, figsize = (8,6), fontsize=12):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d",cmap='Blues')
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


# In[ ]:


prec,recall,threshold = precision_recall_curve(y,val_probs)
pr_auc = metrics.auc(recall,prec)
plt.plot(recall, prec, color='b',label = 'AUC = %0.3f' % pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR-Curve for Discriminatory Network')
plt.legend()
plt.show()


# In[ ]:


cf_matrix = confusion_matrix(y,val_pred)
print_confusion_matrix(cf_matrix,class_names=['nothing','ship'])

