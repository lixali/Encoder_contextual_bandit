import os
import torch.nn as nn
from process import generate_data
import pandas as pd
import pickle
from torch import tensor
import torch
import re
from collections import defaultdict
from transformers import BertModel, BertTokenizer
import json


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# class readPickle(Dataset):

#     def __init__(self, demographics=[], 
#                 lemma=True, stemming = False, stopw = False, keywords=[], news_window = 1, metric = "GOVT",
#                 pickled_news_file=None, start="2020-01-01", end="2022-12-31"):


#     if pickled_news_file is not None:
#         with open(pickled_news_file, 'rb') as f:
#             self.all_news=pickle.load(f)

# demographic_vals = {
#     'SEX': ['1','2'],
#     'MARRY': ['1', '3', '4', '5', ' '],
#     'REGION': ['1', '2', '3', '4'],
#     'EDUC': ['1', '2', '3', '4', '5', '6', ' ']
# }

# def process_demographic(self, demographic, val):
#     one_hot = None
#     one_hot = torch.zeros(len(demographic_vals[demographic]))
#     index = demographic_vals[demographic].index(val)
#     one_hot[index]=1
#     return one_hot

# def getLength(sentence):
#     model = BertModel.from_pretrained('bert-base-uncased')
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#     return len(tokenizer.tokenize(sentence))

def readPickle(picklefile):
    with open(picklefile, 'rb') as f:
        return pickle.load(f)
    
model = BertModel.from_pretrained('bert-base-uncased').to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def getBertEmbedding(picklefile, directory):
            print("processing ", picklefile)
            
            all_news = readPickle(picklefile)
            bert_maxsize = 510
            #print(all_news.keys())
            for month in all_news:
                print(month)
                news = all_news[month][0] #### all_news is a dictionary with key as month and value as a list of tuple (news,date)
                # print(all_news[month])
                # print(news)
                count, sumtotal = 0, 0
                text = ""
                n = len(all_news[month])
                result = []
                for news,_ in all_news[month]:
                    tokenCount = len(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(news)))
                    # tokenizer.convert_tokens_to_ids(tokenized_text)

                    if(sumtotal + tokenCount > bert_maxsize):    
                        marked_text = "[CLS] " + text + " [SEP]"
                        tokenized_text = tokenizer.tokenize(marked_text)
                        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                        segments_ids = [1] * len(tokenized_text[:512])
                        model_ip_ids = torch.tensor([indexed_tokens[:512]]).to(device)
                        model_attention_mask = torch.tensor([segments_ids]).to(device)
                        modelop = model(input_ids = model_ip_ids, attention_mask=model_attention_mask)[1]
                        result.append(modelop.detach().cpu())
                        text = news
                        sumtotal = tokenCount
                        #print(count, n)
                    else:
                        sumtotal += tokenCount
                        text = text + news

                print(picklefile)
                print(len(result))
                question = picklefile.split("_")[3].split(".")[0]
                file_dir = directory + question + "/"
                if not os.path.exists(file_dir):
                    os.makedirs(file_dir)
                filename =  file_dir+month + ".torch"
                rnn_input = torch.stack(result)
                torch.save(rnn_input, filename)

def process_cap(directory):
    for picklefile in os.listdir(directory):
        if not picklefile.endswith(".pickle"):
            continue
        print(picklefile)
        getBertEmbedding(directory+picklefile, directory)

process_cap("data/cap1000/")