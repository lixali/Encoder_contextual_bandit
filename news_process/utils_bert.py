from getNews import getFilteredNews
from process import generate_data
from torch.utils.data import Dataset
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import pickle
from torch import tensor
import torch
import re

class NewsDataset(Dataset):

    #if idx is 0, it gives 2020-01 (indexes can be negative too, in that case we go to 2019)
    #count is the number of months to include starting from idx going up
    demographic_vals = {
        'SEX': ['1','2'],
        'MARRY': ['1', '3', '4', '5', ' '],
        'REGION': ['1', '2', '3', '4'],
        'EDUC': ['1', '2', '3', '4', '5', '6', ' ']
    }

    def process_demographic(self, demographic, val):
        one_hot = None
        one_hot = torch.zeros(len(self.demographic_vals[demographic]))
        index = self.demographic_vals[demographic].index(val)
        one_hot[index]=1
        return one_hot

    def get_month_strings(self, base, count):
        dt = datetime.strptime(base + '-01', '%Y-%m-%d')
        res=[]
        idx = -1*count+1
        while idx<1:
            res.append( (dt + relativedelta(months=idx)).strftime('%Y-%m'))
            idx+=1
        
        return res

    #news_window = 1 means only current month is considered
    #picked_news: when you have the dictionary for a given date range pickled
    #pickle: True when you want to generate news dictionary and pickle it
    def __init__(self, start="", end="", demographics=[], news_window = 1, metric = "GOVT", tensor_dir="cap4"):
        #save imporant parameters
        self.demographics = demographics
        self.news_window = news_window
        self.metric = metric

        
        self.start = start
        self.end = end

        #prepare the survey data
        self.survey_df = generate_data(demographics = demographics, start = self.start, end = self.end)
        
        prev=None
        self.tensor_dict = {}
        for idx in range(self.__len__()):
            base_month = self.survey_df.iloc[idx]['YYYYMM'][:4]+"-"+self.survey_df.iloc[idx]['YYYYMM'][4:6]
            if base_month !=prev:
                prev=base_month
                cur_result=[]
                months = self.get_month_strings(base_month, self.news_window)

                for month in months:
                    file_name = "data/" + tensor_dir +"/"  +metric+"/"+month+".torch"
                    cur_tensor = torch.load(file_name)
                    cur_result.append(cur_tensor)
                
                self.tensor_dict[base_month] = torch.cat(cur_result).squeeze(0)

    def __len__(self):
        return len(self.survey_df)

    def __getitem__(self, idx):
        X = {}
        for demographic in self.demographics:
            X[demographic] = self.process_demographic(demographic, self.survey_df.iloc[idx][demographic])
        
        base_month = self.survey_df.iloc[idx]['YYYYMM'][:4]+"-"+self.survey_df.iloc[idx]['YYYYMM'][4:6]
        
        X["news"] = self.tensor_dict[base_month]
        y = self.survey_df.iloc[idx][self.metric]
        return (X,tensor(float(y)))

def getEmbeddingFromJson(month):

    embedding = torch.load(month)
    return embedding
#teting code

# print(getEmbeddingFromJson("2019-01"))
keywords_dict = {}
keywords_dict["GOVT"] = ["governemnt", "trump", "biden", "election", "president", "congress", "senate", "democrat", "republican", "political party", "political parties", "political ideology", "political beliefs", "political views", "political views", "political system", \
                        "political system", "hilary", "clinton", "bernie", "sanders", "democratic", "republican", "democratic party", "republican party", "democratic nominee", "republican nominee", "democrat", "republican", "dem"]
keywords_dict["HOM"] = ["home", "house", "mortgage", "rent", "interest rates", "housing market", "housing prices", "housing inventory", "housing affordability", "housing supply", "housing demand", "housing shortage", "housing crisis", "housing bubble", "housing crash"]
keywords_dict["PAGO"] = ["jobs", "inflation", "unemployment", "employment", "job market", "job growth", "job creation", "job loss", "job losses", "job openings", "job openings", "job openings", "job openings", "job openings", "job openings", "job openings", "job openings"]    

# NewsDataset(news_window=4)


# obj = NewsDataset(pickled_news_file="data/2019-01-01_to_2022-05-31.pickle", news_window=4)
# print(obj.__len__())
# #exit()
# for i in range(obj.__len__()):
#     X,y = obj.__getitem__(i)
#     # print(X["news"])
#     exit()




