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
import os
import sys
import numpy as np
import csv
import skvideo.io
import collections
import pickle
from reader import readShortVideo
from reader import getVideoList
from data_process2 import *
from p1_util import*
import matplotlib.pyplot as plt
import argparse

read_valid_txt = 0
batch_size = 1

n_label = 11
num_epochs=100
load_model=1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( 'val_path', help='validation video  directory', type=str)
    parser.add_argument( 'label_csv', help='prediction masks directory', type=str)
    parser.add_argument( 'out_dir', help='path of output file', type=str)
    args = parser.parse_args()

    torch.manual_seed(29)
    torch.cuda.manual_seed_all(29)
    np.random.seed(29)
    val_path = args.val_path
    val_csv = args.label_csv
    out_dir = args.out_dir
        
    #train_path = "./HW5_data/TrimmedVideos/video/train/"
    #val_path = "./HW5_data/TrimmedVideos/video/valid/"
    #train_csv = "./HW5_data/TrimmedVideos/label/gt_train.csv"
    #val_csv = "./HW5_data/TrimmedVideos/label/gt_valid.csv"

    if read_valid_txt == 1:
        calculate_acc(valid_csv)
    else:
        #train_data = extract(train_path, train_csv, 1,n_label,batch_size,'train')
        #print(len(train_data))
        val_data = extract(val_path, val_csv, 0,n_label,batch_size,'val')
        print(len(val_data))
        if load_model==0:
            model = training(train_data, val_data, "./loss.jpg")
        else:
            model = training_model()
            model.load_state_dict(torch.load('./p1_1.pkt'))
            model=model.cuda()
        testing(val_data, model ,out_dir)
        #calculate_acc(val_csv, "./p1_valid.txt")
        
