from keras.regularizers import l2
from keras.layers import *
from keras.engine import Layer
from keras.applications.vgg16 import *
from keras.models import *
from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, History, EarlyStopping
from keras.utils.data_utils import get_file
#import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os
import argparse

def trans_masks(files):
    '''
    Read masks from directory and tranform to categorical
    '''

    n_masks = len(files)
    masks = np.empty((n_masks,512, 512,3))

    for i, file in enumerate(files):
        #maxs=np.empty((512,512,7))
        maxi=np.argmax(file,axis=2)
        '''
        for j in range(512):
            for k in range(512):
                maxs[j][k]=np.full((7),maxi[j][k])
        '''
        mask = maxi       
        #mask = 0*mask[:,:,0]+ mask[:, :, 1]+ 2*mask[:, :, 2]+ 3*mask[:, :, 3]+ 4*mask[:, :, 4]+ 5*mask[:, :, 5]+ 6*mask[:, :, 6]
        masks[i, mask == 0] = np.array([0,1,1])  # (Cyan: 011) Urban land 
        masks[i, mask == 1]= np.array([1,1,0])  # (Yellow: 110) Agriculture land 
        masks[i, mask == 2]= np.array([1,0,1])  # (Purple: 101) Rangeland 
        masks[i, mask == 3] = np.array([0,1,0])  # (Green: 010) Forest land 
        masks[i, mask == 4] = np.array([0,0,1])  # (Blue: 001) Water 
        masks[i, mask == 5] = np.array([1,1,1])  # (White: 111) Barren land 
        masks[i, mask == 6] = np.array([0,0,0])  # (Black: 000) Unknown
         
    return masks



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( 'val_path', help='validation pic  directory', type=str)
    parser.add_argument( 'path', help='prediction masks directory', type=str)
    args = parser.parse_args()

    val_path = args.val_path
    path = args.path
    if val_path[-1]!='/':
        val_path=val_path+'/'
    if path[-1]!='/':
        path=path+'/'
    #load validation
    val_files = os.listdir(val_path)
    val_sat=[]
    val_mask=[]
    for file in val_files:
        if file == ".DS_Store":
            val_files.remove(file)
        if "mask" in file:
            
            val_mask.append(file)
        elif "sat" in file:
            val_sat.append(file)

    val_sat.sort()
    val_mask.sort()



    val_pic=[]
    for pic in val_sat:
        val_pic.append(np.array(scipy.misc.imread(val_path + pic)))

    val_pic=np.array(val_pic,dtype=np.uint8)
    nums=[12]
    print('data loaded')
    model=get_file('32s_model-12-0.3627.h5','https://www.dropbox.com/s/6cle4sw5i5bibjt/32s_model-12-0.3627.h5?dl=1',cache_subdir='models')
    model=load_model(model)
    print('model loaded')

    

	

	#predict
    val_predict = model.predict(val_pic, batch_size=12)
    val_predict=trans_masks(val_predict)
    #print(val_predict.shape)
    for j in range(len(val_predict)):
        scipy.misc.imsave(path+'{}_mask.png'.format(str(j).zfill(4)), val_predict[j])
    print('prediction of epoch{} is saved'.format(12))