from netCDF4 import Dataset
import numpy as np
import os
import random
import torch
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler

import h5py


class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay, p=2):

        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        self.weight_info(self.weight_list)

    def to(self, device):
        '''
        :param device: cude or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)  
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        '''
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        '''
        :param weight_list:
        :param p:
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        '''
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")




def dealNaN(data):
    N=0
    for i in range(0,data.shape[0]):
        for j in range(0,data.shape[1]):
            if type(data[i][j])==np.ma.core.MaskedConstant:
                data[i][j]=0   
                N+=1
    return data,N



def trainTestSplit_radom(trainingSet, targetSet, train_size):
    totalNum = int(len(trainingSet))    
    trainIndex = list(range(totalNum))  
    x_train = []  
    y_train = []  
    x_valid = []  
    y_valid = []  
    trainNum = int(totalNum * train_size) 

    for i in range(trainNum):   
        randomIndex = int(random.uniform(0, len(trainIndex)))
        x_train.append(trainingSet[randomIndex])
        y_train.append(targetSet[randomIndex])
        del (trainIndex[randomIndex])  
    for i in range(totalNum - trainNum):   
        x_valid.append(trainingSet[trainIndex[i]])
        y_valid.append(targetSet[trainIndex[i]])
    return x_train, y_train, x_valid, y_valid


def trainTestSplit(trainingSet, targetSet, train_size):
    totalNum = int(len(trainingSet))
    trainIndex = list(range(totalNum))  
    x_train = []  
    y_train = []  
    x_valid = []  
    y_valid = []  
    if train_size == 'last_year':
        trainNum = totalNum - 1200
        print('totalNum', totalNum)
        print('trainNum', trainNum)
    else:
        trainNum = int(totalNum * train_size) 
    for i in range(trainNum):
        x_train.append(trainingSet[i])
        y_train.append(targetSet[i])
    for j in range(trainNum, totalNum):
        x_valid.append(trainingSet[j])
        y_valid.append(targetSet[j])
        # print(np.array(x_train).shape)
    return x_train, y_train, x_valid, y_valid

# Dataset Packaging h5
def cache(fname, x_train, y_train, x_valid, y_valid):
    h5 = h5py.File(fname, 'w')
    h5.create_dataset('x_train', data=x_train)
    h5.create_dataset('y_train', data=y_train)
    h5.create_dataset('x_valid', data=x_valid)
    h5.create_dataset('y_valid', data=y_valid)

# Dataset Packaging h5
def cache_all(fname, data):
    h5 = h5py.File(fname, 'w')
    h5.create_dataset('data', data=data)

def read_cache(fname):
    f = h5py.File(fname, 'r')
    x_train, y_train, x_valid, y_valid = [], [], [], []
    x_train = f['x_train'][()]
    y_train = f['y_train'][()]
    x_valid = f['x_valid'][()]
    y_valid = f['y_valid'][()]
    return x_train, y_train, x_valid, y_valid

# Read h5 dataset
def read_cache_all(fname):
    f = h5py.File(fname, 'r')
    data=[]
    data= f['data'][()]
    return data

# Read NC file
def read_nc(fname,dataname):
    f = Dataset(fname)
    data=f.variables[dataname][0]
    return data

# Data normalization
def data_normal(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    mmn_data = scaler.transform(data)
    return mmn_data

# patch segmentation
def patch_split(datalist,winW,winH):     #[[[21],[21],[21],[21].....21]]
    # print(datalist.shape[0],datalist.shape[1])
    h=datalist.shape[0]
    w=datalist.shape[1]
    new_data = []
    stepSize = 1                                             #datalist.shape=(240,240)
    for i in range(0,datalist.shape[0]-(h-winH*(h//winH)), winH):             #(h-winH*(h//winH))
        for j in range(0, datalist.shape[1]-(w-winW*(w//winW)), winW):         #(w-winW*(w//winW))
            # print(datalist[i:i+winW,j:j+winH])
            data = datalist[i:i + winW, j:j + winH]
            new_data.append(data)
    # print(len(new_data),"-------------")
    # print(np.array(new_data).shape)
    return np.array(new_data)

# Preprocessing
def get_data_h5(path,lat_start,lat_end,lon_start,lon_end):
    nc = read_cache_all(path)
    data = nc[lat_start: lat_end, lon_start: lon_end]
    return data

def get_data_nc(path,dataname,lat_start,lat_end,lon_start,lon_end):
    nc = read_nc(path,dataname)
    data = nc[lat_start: lat_end, lon_start: lon_end]
    return data



def split_sequence(sequence1,sequence2, sliding_window_width,sw_len):
    X, Y = [], []
    print('length',len(sequence1))
    for i in range(len(sequence1)):
        end_element_index1 = i + sliding_window_width
        end_element_index2=end_element_index1+sw_len
        # print(end_element_index)
        if end_element_index1 > len(sequence1): 
            break
        sequence_x = sequence1[i:end_element_index1] 
        sequence_y= sequence2[i:end_element_index1]
        X.append(sequence_x)
        # print(len(sequence_x))
        Y.append(sequence_y)
    return np.array(X), np.array(Y)