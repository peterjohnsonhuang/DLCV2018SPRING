from keras.models import Model
from keras.regularizers import l2
from keras.layers import *
from keras.engine import Layer
from keras.applications.vgg16 import *
from keras.models import *
from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, History, EarlyStopping

import numpy as np
import scipy.misc
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os
import argparse




def FCN_Vgg19_32s(input_shape=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=7):

    img_input = Input(shape=(512,512,3))
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2',kernel_regularizer=l2(0.00005))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2',kernel_regularizer=l2(0.00005))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4',kernel_regularizer=l2(0.00005))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4',kernel_regularizer=l2(0.00005))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4',kernel_regularizer=l2(0.00005))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    
    vgg = Model(img_input, x)

    weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5','https://www.dropbox.com/s/lh7c3zj6v5tk3he/vgg19_weights_tf_dim_ordering_tf_kernels.h5?dl=1',cache_subdir='models')
    
    vgg.load_weights(weights_path, by_name=True)
    
    

    x =  Conv2D( 4096 , ( 2, 2 ) , activation='relu' , padding='same')(x)
    x = Dropout(0.5)(x)
    x =  Conv2D( 4096 , ( 1 , 1 ) , activation='relu' , padding='same',kernel_regularizer=l2(0.00005))(x)
    x = Dropout(0.5)(x)

    x =  Conv2D( classes ,  ( 1 , 1 ) ,activation='linear', padding='valid',kernel_initializer='he_normal',kernel_regularizer=l2(0.00005))(x)
    x = Conv2DTranspose( classes , kernel_size=64 ,  strides=32 , use_bias=False,activation='softmax',padding='same'  )(x)
  
    model = Model( img_input , x)


    return model
def read_masks(files,filepath):
    '''
    Read masks from directory and tranform to categorical
    '''

    n_masks = len(files)
    masks = np.empty((n_masks, 512, 512,7))

    for i, file in enumerate(files):
        mask = scipy.misc.imread(os.path.join(filepath, file))
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[i, mask == 3] = np.array([1,0,0,0,0,0,0])  # (Cyan: 011) Urban land 
        masks[i, mask == 6] = np.array([0,1,0,0,0,0,0])  # (Yellow: 110) Agriculture land 
        masks[i, mask == 5] = np.array([0,0,1,0,0,0,0])  # (Purple: 101) Rangeland 
        masks[i, mask == 2] = np.array([0,0,0,1,0,0,0])  # (Green: 010) Forest land 
        masks[i, mask == 1] = np.array([0,0,0,0,1,0,0])  # (Blue: 001) Water 
        masks[i, mask == 7] = np.array([0,0,0,0,0,1,0])  # (White: 111) Barren land 
        masks[i, mask == 0] = np.array([0,0,0,0,0,0,1])  # (Black: 000) Unknown
        masks[i, mask == 4] = np.array([0,0,0,0,0,0,1])  # (Red: 100) Unknown
    return masks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( 'train_path', help='training pic  directory', type=str)
    parser.add_argument( 'val_path', help='validation masks directory', type=str)
    args = parser.parse_args()

    train_path = args.train_path
    val_path = args.val_path
    if val_path[-1]!='/':
        val_path=val_path+'/'
    if train_path[-1]!='/':
        train_path=train_path+'/'

    fcn32_model=FCN_Vgg19_32s()

    fcn32_model.summary()

    optimizer=Adam(lr=0.0001)
    fcn32_model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])






    train_files = os.listdir(train_path)
    train_sat=[]
    train_mask=[]
    for file in train_files:
        if file == ".DS_Store":
            train_files.remove(file)
        if "mask" in file:
        
            train_mask.append(file)
        elif "sat" in file:
            train_sat.append(file)



    train_sat.sort()
    train_mask.sort()


    print('train load complete')

    train_pic=[]
    train_mask7=read_masks(train_mask,filepath=train_path)
    #train_mask7=mask_tansform7(train_pic2)
    print('mask transform complete')
    for name in train_sat:
        train_pic.append(np.array(scipy.misc.imread(train_path + name)))



    train_pic=np.array(train_pic,dtype=np.uint8)
    train_mask7=np.array(train_mask7,dtype=np.uint8)


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
    val_mask7=read_masks(val_mask,filepath=val_path)
    #val_mask7=mask_tansform7(val_pic2)
    for name in val_sat:
        val_pic.append(np.array(scipy.misc.imread(val_path + name)))
    val_pic=np.array(val_pic,dtype=np.uint8)
    val_mask7=np.array(val_mask7,dtype=np.uint8)


    mode = 'vgg19v4'
    print('training with mode '+mode)
    checkpointer = ModelCheckpoint(filepath=mode+'_model-{epoch:02d}-{val_loss:.4f}.h5', verbose=0, save_best_only=True, period=2)
    # history = History()
    earlystopping = EarlyStopping(patience=10, min_delta=0.00)
    fcn32_model.fit(train_pic,train_mask7, batch_size=16, epochs=30, verbose=1,validation_data=(val_pic, val_mask7), callbacks=[checkpointer, earlystopping])

