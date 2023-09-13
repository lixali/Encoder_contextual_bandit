
### so the question is why I do not just 
### calculate the total reward in each column and then just pick the column with largest value?
import pandas as pd
import random

random.seed(1000)
df = pd.read_csv("Ads_Optimisation.csv")

def randomSelection(df):

    values = df.values


    rowNum = 10000
    colNum = 10

    totalReward = 0
    for i in range(rowNum):

        randomCol = random.randrange(colNum) ### the random number will not exceed colNum
        totalReward += values[i, randomCol]

    return totalReward

print(randomSelection(df))
