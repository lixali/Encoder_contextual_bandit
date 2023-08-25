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
from math import log2

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
    def __init__(self, demographics=[], 
                lemma=True, stemming = False, stopw = False, keywords=[], news_window = 1, metric = "GOVT",
                pickled_news_file=None, start="2020-01-01", end="2022-12-31", vocab = None, df_dict=None, idf=True):
        #save imporant parameters
        self.demographics = demographics
        self.news_window = news_window
        self.metric = metric

        if pickled_news_file is not None:
            regex = r'\d{4}-\d{2}-\d{2}'
            date_strings = re.findall(regex, pickled_news_file)
            self.start = date_strings[0]
            self.end = date_strings[1]
        else:
            self.start = start
            self.end = end

        #prepare the survey data
        self.survey_df = generate_data(demographics=demographics, start=self.start, end=self.end) 

        self.vocab = {}     

        #if news pickle file name has been provided
        if pickled_news_file is not None:
            with open(pickled_news_file, 'rb') as f:
                self.all_news=pickle.load(f)
        else:
            date_obj = datetime.strptime(self.start, '%Y-%m-%d')
            new_date_obj = date_obj - relativedelta(years=1)
            actual_start = new_date_obj.strftime('%Y-%m-%d')

            self.all_news = getFilteredNews(actual_start, end, lemma=lemma, stemming = stemming, stopw = stopw, jsonFile = "data/newsAll.json", keywords = keywords)
            if keywords == []:
                with open("data/"+self.start+"_to_"+self.end+".pickle", 'wb') as f:
                    pickle.dump(self.all_news, f)
            elif keywords != []:
                with open("data/"+self.start+"_to_"+self.end+"_" + keywords[0] + ".pickle", 'wb') as f:
                    pickle.dump(self.all_news, f)
        
        #if vocab is None then build the vocab (train) else use the vocab (test)
        v_index = 0
        # use this flag to know if we are dealing with train or test 
        train=False
        if vocab is None:
            train=True
            for k,v in self.all_news.items():
                for article,_ in v:
                    for word in article.split():
                        if word not in self.vocab:
                            self.vocab[word]=v_index
                            v_index+=1
        else:
            self.vocab = vocab
        
        prev=None

        #df_dict records the number of documents each word appears in
        #if it is a trainig data, we construct df_dict, for test we just use the one from training data
        if train:
            self.df_dict = [0]*len(self.vocab)
        else:
            self.df_dict = df_dict

        #tf_dict records the term frequencies list for each document
        self.tf_dict = {}
        for idx in range(self.__len__()):
            base_month = self.survey_df.iloc[idx]['YYYYMM'][:4]+"-"+self.survey_df.iloc[idx]['YYYYMM'][4:6]
            if base_month !=prev:
                prev=base_month

                months = self.get_month_strings(base_month, self.news_window)

                #current_counts tracks the term frequencies for each word in curent doc
                current_counts = [0]*len(self.vocab)
                local_dict = {}
                for month in months:
                    for article,_ in self.all_news[month]:
                        for word in article.split():
                            if not train and word in self.vocab:
                                current_counts[self.vocab[word]]+=1
                            # we only update df_dict if it is training data, for test data, we use df dict from train dataloader
                            if word not in local_dict and train:
                                self.df_dict[self.vocab[word]]+=1
                                local_dict[word]=1
            
                self.tf_dict[base_month]=current_counts
        
        #apply tf idf
        N = len(self.tf_dict)
        for k,v in self.tf_dict.items():
            for i in range(len(self.vocab)):
                if self.df_dict[i]!=0:
                    self.tf_dict[k][i]= self.tf_dict[k][i]*log2(N/self.df_dict[i])
    
    def __len__(self):
        return len(self.survey_df)

    def __getitem__(self, idx):
        X = {}
        for demographic in self.demographics:
            X[demographic] = self.process_demographic(demographic, self.survey_df.iloc[idx][demographic])

        base_month = self.survey_df.iloc[idx]['YYYYMM'][:4]+"-"+self.survey_df.iloc[idx]['YYYYMM'][4:6]
        X["news"] = torch.tensor(self.tf_dict[base_month])

        y = self.survey_df.iloc[idx][self.metric]
        return (X,tensor(float(y)))

#teting code
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




