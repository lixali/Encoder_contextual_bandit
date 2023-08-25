### write a function that look into data folder and third_eye_news folder and return a list of news title sorted by date; the function signature takes time as input
### for new york times news, I can either extract the "headline", "abstract" or I can extract "lead_paragraph"; it seems that "lead_paragraph " contains a lot more information for new york times
### "pub_date"

import json
import nltk
import os
from datetime import datetime
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re



def is_timestamp_format(s):
    pattern = r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$'
    if re.match(pattern, s):
        return True
    else:
        return False

def processTokens(sentence, lemma=True, stemming = False, stopw = False):

    tokens = word_tokenize(sentence)

    # Remove stop words
    if stopw:
        stop_words = set(stopwords.words('english'))
        tokens = [token.lower() for token in tokens if token.lower() not in stop_words]

    # Stem the tokens
    if stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token.lower()) for token in tokens]

    # Lemmatize the tokens
    if lemma:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]


    return " ".join(tokens)

def getAllNews(): 
    i = 0
    j = 0
    new_york_times = './data/'
    third_eye_news = './third_eye_news/'
    newsDict = defaultdict(list)
    newsList = []

    for filename in os.listdir(new_york_times):
        
        if filename.endswith('.json'):
            file_path = os.path.join(new_york_times, filename)
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                # Do something with the JSON data here

                for news in json_data["response"]["docs"]:
                    currNewsSummary = news["lead_paragraph"]
                    # currNewsSummary = processTokens(currNewsSummary)
                    pub_date = news["pub_date"]
                    dt = datetime.strptime(pub_date, '%Y-%m-%dT%H:%M:%S%z')
                    pub_date = dt.strftime('%Y-%m-%d')
                    if i < 1:
                        print(file)
                        # print(news)
                        print((currNewsSummary, pub_date))
                        

                    # newsList.append((currNewsSummary, pub_date))
                    # newsList = sorted(newsList, key=lambda d: datetime.strptime(d[1], '%Y-%m-%d'))

                    newsDict[pub_date].append(currNewsSummary)
                    i += 1

    for filename in os.listdir(third_eye_news):
        if filename.endswith('.txt'):
            file_path = os.path.join(third_eye_news, filename)
            with open(file_path, 'r') as file:
                next(file)
                for line in file:
                    fields = line.strip().split('\t')
                    currNewsSummary = fields[-1]
                    currNewsSummary = currNewsSummary.replace('\n', '').replace('\\', '')
                    # currNewsSummary = processTokens(currNewsSummary)
                    pub_date = fields[0]
                    
                    if is_timestamp_format(pub_date):
                        pub_date = datetime.strptime(pub_date, "%Y-%m-%d %H:%M:%S")
                        pub_date = pub_date.strftime("%Y-%m-%d")
                        
                    else: continue
                    if j == 0:
                        print(file)
                        print((currNewsSummary, pub_date))
                    newsDict[pub_date].append(currNewsSummary)

                    j += 1


    newsDict = {key: newsDict[key] for key in sorted(newsDict)}
    with open('data/newsAll.json', 'w') as f:
    # Use the json.dump() function to write the dictionary to the file
        json.dump(newsDict, f)

def contains_keyword(s, keywords):
    s = s.lower()
    for keyword in keywords:
        if keyword.lower() in s:
            return True
    return False

def getFilteredNews(startDate, endDate, lemma=True, stemming = False, stopw = False, jsonFile = "data/newsAll.json", keywords = []):
    # print("################## starting to read news ##################")
    finalOutput = {}
    with open(jsonFile, 'r') as f:
        data = json.load(f)
    
    # print("################## finished reading news ##################")
    for newsDate in data:
        if startDate <= newsDate <= endDate:
            key = newsDate[:-3]
            print(key)
            sentences = data[newsDate]
            for sentence in sentences:
                
                if len(keywords) != 0: ### there are keywords that needs to be filtered
                    if contains_keyword(sentence, keywords):
                        sentence = processTokens(sentence, lemma=lemma, stemming = stemming, stopw = stopw)
                        if(key not in finalOutput):
                            finalOutput[key]=[]
                        finalOutput[key].append((sentence, newsDate))
                else: ### get every news that is in between starting and ending date
                    sentence = processTokens(sentence, lemma=lemma, stemming = stemming, stopw = stopw)
                    if(key not in finalOutput):
                        finalOutput[key]=[]
                    finalOutput[key].append((sentence, newsDate))
    return finalOutput

# getAllNews()

#finalOutput = getFilderedNews("2020-01-01", "2020-01-02", keywords = ["TRUMP", "inflation"])
#print(finalOutput)


    
