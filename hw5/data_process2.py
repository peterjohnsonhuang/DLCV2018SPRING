import os
import sys
import numpy as np
import csv
import pickle
from reader import readShortVideo,readShortVideo2
from reader import getVideoList

import torch
import argparse
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import scipy.misc


def extract(folder, csvpath, load,num_class,batch_size,name):
    print("extract frames...")
    
    frames = []
    labels = []
    video_list = getVideoList(csvpath)
    
    if (load == 0):
        for i in range(len(video_list["Video_name"])):
            frame = readShortVideo(folder, video_list["Video_category"][i],video_list["Video_name"][i])
            frame = np.mean(frame,axis=0,keepdims=True)
            #print(frame.shape)
            for j in range(len(frame)):
                frames.append(np.moveaxis(frame[j], -1, 0))
                label = np.zeros(num_class)
                label[int(video_list["Action_labels"][i])] = 1
                labels.append(label)
        frames=np.array(frames,dtype=np.uint8)
        labels=np.array(labels,dtype=np.uint8)
        #np.save("./"+name+"_frames.npy",frames)
        #np.save("./"+name+"_labels.npy",labels)
    elif load == 1:
        frames=np.load("./"+name+"_frames.npy")
        labels=np.load("./"+name+"_labels.npy")
    print(frames.shape,labels.shape)
    data = [(frames[i], labels[i]) for i in range(len(frames))]
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return dataloader


def extract2(folder, csvpath, load,num_class,batch_size,name):
    print("extract frames...")
    
    frames = []
    labels = []
    video_list = getVideoList(csvpath)
    
    if (load == 0):
        for i in range(len(video_list["Video_name"])):
            frame = readShortVideo2(folder, video_list["Video_category"][i],video_list["Video_name"][i])
            #frame = np.mean(frame,axis=0,keepdims=True)
            #print(frame.shape)
            for j in range(len(frame)):
                frames.append(np.moveaxis(frame[j], -1, 0))
                label = np.zeros(num_class)
                label[int(video_list["Action_labels"][i])] = 1
                labels.append(label)
        frames=np.array(frames,dtype=np.uint8)
        labels=np.array(labels,dtype=np.uint8)
        #np.save("./"+name+"_frames2_16.npy",frames)
        #np.save("./"+name+"_labels2_16.npy",labels)
    elif load == 1:
        frames=np.load("./"+name+"_frames2_16.npy")
        labels=np.load("./"+name+"_labels2_16.npy")
    print(frames.shape,labels.shape)
    
    
    data = [(frames[i], labels[i]) for i in range(len(frames))]
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return dataloader
    


def extract3(folder, txtfolder, load,num_class,batch_size,data_name):
    names=os.listdir(folder)


    print("extract frames...")
    
    frames = []
    labels = []


    if (load == 0):
        for name in names:
            all_frames=os.listdir(folder+name)
            frames_num=len(all_frames)
            #print(frames_num)
            label_file = open(txtfolder+name+'.txt', "r")
            true=[]
            for line in label_file:
                if line == '\n':
                    continue
                true.append(int(line[:-1]))
            cnt=0
            for i in range(frames_num):
                
                if i%int(frames_num/batch_size)==0:
                    #print(np.moveaxis(np.array(scipy.misc.imread(folder+name+'/'+all_frames[i])),-1,0).shape)
                    frames.append(np.moveaxis(np.array(scipy.misc.imread(folder+name+'/'+all_frames[i])),-1,0))
                    label = np.zeros(num_class)
                    label[int(true[i])] = 1
                    labels.append(label)
                    cnt+=1
                    if cnt==batch_size:
                        break
            
            while cnt<batch_size :
                print(cnt)
                frames.append(frames[-1])
               
                labels.append(labels[-1])
                cnt+=1
        print(len(frames))
        frames=np.array(frames,dtype=np.uint8)
        labels=np.array(labels,dtype=np.uint8)
        np.save("./"+data_name+"_frames3_256.npy",frames)
        np.save("./"+data_name+"_labels3_256.npy",labels)
    elif load == 1:
        frames=np.load("./"+data_name+"_frames3_256.npy")
        labels=np.load("./"+data_name+"_labels3_256.npy")
    print(frames.shape,labels.shape)
    
    
    data = [(frames[i], labels[i]) for i in range(len(frames))]
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return dataloader,labels
    