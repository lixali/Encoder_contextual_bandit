import os
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_log_error
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import random
random.seed(577)

import numpy as np
np.random.seed(577)

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import mean_squared_error, r2_score
from arch import *
import os
from utils_bow import *
# print("Current working directory: ", os.getcwd())
# print(os.listdir("."))
# import sys
# sys.path.append("..")
from getNews import getFilteredNews

torch.set_default_tensor_type(torch.FloatTensor)
# torch.use_deterministic_algorithms(True)
torch.manual_seed(577)
torch_device = torch.device("cpu")

import torch

class MSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return torch.mean((torch.log1p(y_pred) - torch.log1p(y_true))**2)


class CustomLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        diff = torch.abs(y_pred - y_true)
        loss = -torch.log(1 - torch.min(torch.tensor([1.0]), diff))
        return torch.mean(loss)


if __name__ == "__main__":
    demographics = ['SEX', 'MARRY', 'REGION', 'EDUC']
    demographics = ['SEX']
    dataloader = NewsDataset(pickled_news_file='data/2020-01-01_to_2022-05-31.pickle', news_window=2,
                             demographics=demographics, vocab=None, idf=False, metric='PAGO')
    print("loaded training data")
    test_dataloader = NewsDataset(pickled_news_file='data/2022-06-01_to_2022-12-31.pickle', news_window=2,
                                demographics=demographics, vocab=dataloader.vocab, df_dict=dataloader.df_dict, idf=False, metric='PAGO')
    
    print("loaded testing data")
    #dataloader = NewsDataset(start="2020-01-01", end="2022-05-31", news_window=2)
    #test_dataloader = NewsDataset(start="2022-06-01", end="2022-12-31", news_window=2)

    # print(newsData)
    model = ANN(input_dim = len(dataloader.vocab), hidden_dim= 15, output_dim= 1, demographics = demographics)
    lr = 0.001
    num_epochs = 20


    criterion = nn.MSELoss()
    # criterion = CustomLoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)

    for epoch in range(num_epochs):
        
        true_label = []    
        predicted_label = []
        for i in range(dataloader.__len__()):
            X,y = dataloader.__getitem__(i)
            true_label.append(y.item())
            optimizer.zero_grad()
            outputs = model.forward(X["news"], [X[demographic] for demographic in demographics])
            predicted_label.append(outputs[0].item())
            loss = criterion(outputs.squeeze(), y.float()) ## missing label here
            loss.backward()
            optimizer.step()
            '''
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch+1, num_epochs, i+1, dataloader.__len__(), loss.item()))
            '''
        print("training error : ")
        print(mean_squared_error(true_label, predicted_label))    
       
        true_label = []    
        predicted_label = []
        for i in range(test_dataloader.__len__()):
            X,label = test_dataloader.__getitem__(i)
            true_label.append(label.item())
            o = model(X["news"], [X[demographic] for demographic in demographics])
            predicted_label.append(o[0].item())
        print("testing error : ")
        print(mean_squared_error(true_label, predicted_label))
        
            
        #print("mean squared error : ", mean_squared_error(true_label, predicted_label))
