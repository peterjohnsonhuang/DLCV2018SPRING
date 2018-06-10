import matplotlib
matplotlib.use('Agg')
import torch
import argparse
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.transforms.functional as F
import scipy.misc
import os
import sys
import numpy as np
import h5py
import csv
import skvideo.io
import skimage.transform
import collections
import pickle
from reader import readShortVideo
from reader import getVideoList
from p1 import training_model
from data_process2 import *
from p2_util import *
import matplotlib.pyplot as plt

import argparse

read_feature = 0


batch_size = 16
test = 1
n_label = 11
hidden_size=512




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( 'val_path', help='validation video  directory', type=str)
    parser.add_argument( 'label_csv', help='prediction masks directory', type=str)
    parser.add_argument( 'out_dir', help='path of output file', type=str)
    args = parser.parse_args()
    val_path = args.val_path
    val_csv = args.label_csv
    out_dir = args.out_dir

    torch.manual_seed(29)
    np.random.seed(29)
    torch.cuda.manual_seed_all(29)

    #train_path = "./HW5_data/TrimmedVideos/video/train/"
    #val_path = "./HW5_data/TrimmedVideos/video/valid/"
    #train_csv = "./HW5_data/TrimmedVideos/label/gt_train.csv"
    #val_csv = "./HW5_data/TrimmedVideos/label/gt_valid.csv"
    output_filename = "./p2_result.txt"
    train_feature_file = "./train_feature.npy"
    val_feature_file = "./valid_feature.npy"


    if test == 0:
        if read_feature == 1:
            print("read feature...")
            train_features = read_feature_from_file(train_csv, train_feature_file)
            val_features = read_feature_from_file(val_csv, val_feature_file)
 
        else:
            print("produce feature...")
            train_data = extract2(train_path, train_csv, 1,n_label,batch_size,'train')
            val_data = extract2(val_path, val_csv, 1,n_label,batch_size,'val')
            print("load p1 model...")
            model_p1 = training_model()
            model_p1.load_state_dict(torch.load('./p1_1.pkt'))
            

            model_p1 = model_p1.cuda()
            train_features = get_feature(train_data, model_p1, train_csv, train_feature_file)
            val_features = get_feature(val_data, model_p1, val_csv, val_feature_file)
        print("construct RNN model...")
        
        model_RNN = RNN_model().cuda()
        model_RNN = training(train_features, val_features, model_RNN, "./p2_loss.jpg", output_filename)
        testing(val_features, model_RNN,out_dir)
        calculate_acc(val_csv, output_filename)
        
    else:
        val_data=val_data = extract2(val_path, val_csv, 0,n_label,batch_size,'val')
        model_p1 = training_model()
        model_p1.load_state_dict(torch.load('./p1_1.pkt'))
        model_p1 = model_p1.cuda()
        val_features = get_feature(val_data, model_p1, val_csv, val_feature_file)

        model_RNN = RNN_model().cuda()
        model_RNN.load_state_dict(torch.load('./p2_47.pkt'))
        testing(val_features, model_RNN,out_dir)
        calculate_acc(val_csv, output_filename)