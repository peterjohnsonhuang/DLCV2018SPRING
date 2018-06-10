import matplotlib
matplotlib.use('Agg')
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.transforms.functional as F
import numpy as np
import csv
import matplotlib.pyplot as plt
import os
import sys
from reader import readShortVideo
from reader import getVideoList

read_valid_txt = 0
batch_size = 4

n_label = 11
num_epochs=100


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
        for i in range(x.shape[0]):
            input = x[i]
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

        out = self.fc1(avg_feature)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

def training(data_loader, valid_dataloader, loss_filename):
    print("start training")
    model = training_model().cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    all_loss = []
    all_acc = []
    acc=0
    for epoch in range(num_epochs):
        idx = 0
        train_loss = 0.0
        for data in data_loader:
            img = data[0].type(torch.FloatTensor)
            true_label = data[1].type(torch.FloatTensor)
            true_label = true_label[0].view(1, n_label)
            img = Variable(img).cuda()
            true_label = Variable(true_label).cuda()

            predict_label = model(img)

            loss = nn.BCELoss()(predict_label, true_label)
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, train_loss))
        
        all_loss.append(loss.item())
        temp_acc=testing(valid_dataloader, model)
        all_acc.append(temp_acc)
        if temp_acc>acc:
            acc=temp_acc
            torch.save(model.state_dict(), './p1_1.pkt')
    plot_loss(all_loss, loss_filename)
    plot_acc(all_acc, "./p1_acc.jpg")
    return model

def testing(data_loader, model,out_path):
    num_data = 0
    correct = 0
    save_filename = out_path+'p1_valid.txt'
    
    try:
        os.remove(save_filename)
    except OSError:
        pass
    
    file = open(save_filename, "a")

    for data in data_loader:
        img = data[0].type(torch.FloatTensor)
        true_label = data[1].type(torch.FloatTensor)
        true_label = true_label[0].view(1, n_label)
        img = Variable(img).cuda()
        true_label = Variable(true_label).cuda()
        predict_label = model(img)
        predict_label = np.array(predict_label.data)
        true_label = np.array(true_label.data)
        #print(predict_label.shape,true_label.shape)
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
    print("len of true labels: " + str(len(labels)))
    print("len of predict labels: " + str(len(predict)))
    for i in range(len(predict)):
        if int(labels[i]) == int(predict[i]):
            correct += 1
    file.close()
    print("acc score: " + str(float(correct) / len(predict)))