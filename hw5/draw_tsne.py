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
import matplotlib.pyplot as plt
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
import pandas as pd
from HW5_data.reader import readShortVideo2
from HW5_data.reader import getVideoList
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
from sklearn.manifold import TSNE
from p2_util import*
val_csv = "./HW5_data/TrimmedVideos/label/gt_valid.csv"
val_feature_file = "./valid_feature.npy"

p1_feature="./valid_feature.npy"
valid_features=np.load(p1_feature)
#valid_features=torch.from_numpy(valid_features)

# CNN feature # CNN f 
print(valid_features.shape)


valid_features=np.mean(valid_features,axis=1)
'''
for seq_feature in valid_features:
    CNN_features.append(np.mean(seq_feature,axis=0))
'''
CNN_features = np.array(valid_features)
print(CNN_features.shape)








val_features = read_feature_from_file(val_csv, val_feature_file)
model = RNN_model().cuda()
model.load_state_dict(torch.load("./p2_47.pkt"))

#model.eval()
RNN_feautures = []
valid_y=[]
for data in val_features:
	feature = data[0].type(torch.FloatTensor)
	true_label = data[1].type(torch.FloatTensor)
	feature = Variable(feature).cuda()
	true_label = Variable(true_label).cuda()
	predict_label,out = model(feature)
	hidden=out
	for i in hidden:
		RNN_feautures.append(np.array(i.cpu().data))
	for i in true_label:
		valid_y.append(np.argmax(np.array(i)))

#valid_y=np.array(valid_y)
np.reshape(valid_y,(1,517))
#print(valid_y)
CNN_features_2d  = TSNE(n_components=2, perplexity=40.0, random_state=38).fit_transform(CNN_features)

cm  =  plt.cm.get_cmap("tab20", 11)
plt.figure(figsize=(10,5))
plt.scatter(CNN_features_2d[:,0], CNN_features_2d[:,1], c=valid_y, cmap=cm)
plt.colorbar(ticks=range(11))
plt.clim(-0.5, 10.5)
plt.savefig("CNN_tsne.png")
plt.show()

#print(RNN_feautures)
RNN_feautures = np.array(RNN_feautures)
print(RNN_feautures.shape)

RNN_feautures_2d = TSNE(n_components=2, perplexity=30.0, random_state=38).fit_transform(RNN_feautures)
#print(valid_y.shape)
cm = plt.cm.get_cmap("tab20", 11)
plt.figure(figsize=(10,5))
plt.scatter(RNN_feautures_2d[:,0], RNN_feautures_2d[:,1], c=valid_y, cmap=cm)
plt.colorbar(ticks=range(11))
plt.clim(-0.5, 10.5)
plt.savefig("RNN_tsne.png")
plt.show()


print("Look up table for class color and genres:" )
action = ["Other","Inspect/Read","Open","Take","Cut","Put","Close","Move Around", "Divide/Pull", "Pour", "Transfer"]
pd.DataFrame({"genres":action}, index=list(range(11)))