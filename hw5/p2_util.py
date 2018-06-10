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
from reader import readShortVideo2
from reader import getVideoList
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz


batch_size = 8
frame_num = 16
test = 0
n_label = 11
epochs=200

class training_model(nn.Module):
    def __init__(self):
        super(training_model, self).__init__()
        self.pretrained = torchvision.models.resnet50(pretrained=True)
        self.pretrained.fc = nn.Linear(16 * 32 * 32, 4096)
        self.fc1 = nn.Linear(4096,1024)
        self.fc2 = nn.Linear(1024, n_label)
        self.softmax = nn.Softmax(1)

    def output_feature(self, x):
        features = torch.zeros((x.shape[0], x.shape[1], 4096))
        #print(x.shape)
        for i in range(x.shape[0]):
            input = x[i]
            input.unsqueeze_(0)
            feature = self.pretrained(input)

            
            feature = Variable(feature).cuda()
            features[i] = feature

        return(features)

    def forward(self, x):
        x = self.pretrained(x)
        avg_feature = np.mean(np.array(x.data), axis = 0)
        avg_feature = np.reshape(avg_feature, (1, 4096))
        avg_feature = torch.from_numpy(avg_feature)
        avg_feature = torch.squeeze(avg_feature, 1)
        
        avg_feature = Variable(avg_feature).cuda()

        y = self.fc1(avg_feature)
        y = self.fc2(y)
        y = self.softmax(y)
        return y





def weights_init(m):
    for name, param in m.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 1.0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=1)

class RNN_model(nn.Module):
    def __init__(self):
        super(RNN_model, self).__init__()
        self.hidden_size = 512

        self.rnn = nn.LSTM(4096, 512, 1, dropout=0, bidirectional=True)
        self.drop1 = nn.Dropout(p=0.3)
        self.bn0 = nn.BatchNorm1d(self.hidden_size*2)
        self.fc1 = nn.Linear(512*2 , 256)
        self.bn1 = nn.BatchNorm1d(int(self.hidden_size/2))
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64,11)
        self.softmax = nn.Softmax(1)

    def step(self, input, hidden=None):
        h0 = torch.zeros(2, input.size(1), self.hidden_size) 
        c0 = torch.zeros(2, input.size(1), self.hidden_size)
        
        h0 = h0.cuda()
        c0 = c0.cuda()
        output, hidden = self.rnn(input, hidden)
        output = output[:, -1, :]
        return output, hidden

    def forward(self, inputs, hidden=None, steps=0):
        #print(inputs.shape)
        #if steps == 0: steps = len(inputs[1])
        output, (hn,cn) = self.step(inputs, hidden)
        #print(len(hn))
        #print(output,hn[-1])
        out=output
        #output = self.drop1(output)
        #output = self.bn0(hn[-1])
        output = self.fc1(output)
        #output = self.bn1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.softmax(output)
        return output, out

def training(data_loader, valid_dataloader, model, loss_filename, output_filename):
    print("start training")
    #print(len(data_loader))
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    all_loss = []
    all_acc = []
    acc = 0
    for epoch in range(epochs):
        idx = 0
        train_loss = 0.0
        for data in data_loader:
            #print(data[0].shape,data[1].shape)
            cnn_feature = data[0].type(torch.FloatTensor)
            true_label = data[1].type(torch.FloatTensor)
            true_label = true_label.view(-1, n_label)
            cnn_feature = Variable(cnn_feature).cuda()
            true_label = Variable(true_label).cuda()
            predict_label, hidden = model(cnn_feature, None)
            if true_label.shape[0]!=predict_label.shape[0]:
                predict_label=predict_label[0:true_label.shape[0]]
            #print(predict_label.shape,true_label.shape)
            loss = nn.BCELoss()(predict_label, true_label)
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, epochs, train_loss))
        
        all_loss.append(loss.item())
        temp_acc=testing(valid_dataloader, model)
        all_acc.append(temp_acc)
        if temp_acc>acc:
            acc=temp_acc
            torch.save(model.state_dict(), './p2.pkt')

    plot_loss(all_loss, loss_filename)
    plot_acc(all_acc, "./p2_acc.jpg")
    return model

def testing(data_loader, model,out_path):
    num_data = 0
    correct = 0
    save_filename = out_path+'p2_result.txt'
    
    try:
        os.remove(save_filename)
    except OSError:
        pass
    
    file = open(save_filename, "a")
    for data in data_loader:
        #print(data)
        feature = data[0].type(torch.FloatTensor)
        true_label = data[1].type(torch.FloatTensor)
        feature = Variable(feature).cuda()
        true_label = Variable(true_label).cuda()
        predict_label,hidden = model(feature)
        predict_label = np.array(predict_label.data)
        #avg_pred=np.mean(predict_label,axis=0,keepdims=True)
        true_label = np.array(true_label.data)
        if true_label.shape[0]!=predict_label.shape[0]:
            predict_label=predict_label[0:true_label.shape[0]]

        #print(avg_pred)
        #print(avg_pred.shape,true_label.shape)
        pred = np.argmax(predict_label, 1)
        label = np.argmax(true_label, 1)
        for i in range(len(pred)):
            file.write(str(pred[i]))
            file.write('\n')
            if pred[i] == label[i]:
                correct += 1
        
        num_data += predict_label.shape[0]

    file.write('\n')
    file.close()

    print("test score: " + str(float(correct) / float(num_data)))
    return float(correct) / float(num_data)

def get_feature(data_loader, model, csvpath, output_filename,save=0):
    print("get feature...")
    features = np.zeros((1, frame_num,  4096))
    #print(len(data_loader))
    for i, data in enumerate(data_loader):
        if i % 100 == 0:
            print(i)
        #print (data[0].shape)
        img=data[0].type(torch.FloatTensor)
        #img = torch.from_numpy(data[0])
        img = Variable(img).cuda()
        #print(img)
        #print(img.shape)
        outputs = model.output_feature(img)
        #print(outputs.shape)
        outputs = outputs.data.cpu().numpy()
        outputs = np.reshape(outputs, (frame_num, 3,  4096))
        #print('out:',outputs.shape)
        outputs = np.mean(np.array(outputs), axis = 0,keepdims=True)
        #outputs = np.swapaxes(outputs,0,1)
        #print('out:',outputs.shape)
        #print(outputs.shape)
        if i == 0:
            features = outputs
        else:
            features = np.append(features, outputs, axis=0)
    print(features.shape)
    video_list = getVideoList(csvpath)
    labels = video_list["Action_labels"]
    one_hot_labels = []

    for i in range(len(labels)):

        label = np.zeros(n_label)
        label[int(video_list["Action_labels"][i])] = 1
        
        one_hot_labels.append(label)
    one_hot_labels=np.array(one_hot_labels)
    features=np.array(features)
    print(len(features),len(one_hot_labels))
    print(features.shape,one_hot_labels.shape)

    data = [(features[i], one_hot_labels[i]) for i in range(len(features))]
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    
    try:
        os.remove(output_filename)
    except OSError:
        pass

    #print("start produce feature .h5")
    if save==1:
        np.save(output_filename,features)
    '''
    f = h5py.File(output_filename, 'w')
    f.create_dataset('features', data=features)
    '''
    return dataloader

def read_feature_from_file(csvpath, filename):
    
    features = np.load(filename)
    video_list = getVideoList(csvpath)
    labels = video_list["Action_labels"]

    one_hot_labels = []
    for i in range(len(labels)):
        label = np.zeros(n_label)
        label[int(video_list["Action_labels"][i])] = 1
        one_hot_labels.append(label)
    print("len of feature: " + str(features.shape[0]))
    print("feature size: " + str(features[0].shape))
    data = [(features[i], one_hot_labels[i]) for i in range(len(features))]
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return dataloader


def plot_loss(all_loss, filename):


    
    fig=plt.figure(figsize=(10, 10))
    t = np.arange(0.0, len(all_loss), 1.0)
    line, = plt.plot(t, all_loss, lw=2)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('loss')

    plt.savefig(filename)
    plt.close()
def plot_acc(all_acc, filename):


    
    fig=plt.figure(figsize=(10, 10))
    t = np.arange(0.0, len(all_acc), 1.0)
    line, = plt.plot(t, all_acc, lw=2)
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.title('acc')

    plt.savefig(filename)
    plt.close()

def calculate_acc(csvpath, output_filename):
    print("calculate acc from txt")
    labels = []
    predict = []
    correct = 0
    video_list = getVideoList(csvpath)
    labels = video_list["Action_labels"]
    file = open(output_filename, "r")
    for line in file:
        if line == '\n':
            continue
        predict.append(int(line[:-1]))
    print("num of true labels: " + str(len(labels)))
    print("num of  predict labels: " + str(len(predict)))
    for i in range(len(predict)):
        if int(labels[i]) == int(predict[i]):
            correct += 1
    file.close()
    print("acc score: " + str(float(correct) / len(predict)))