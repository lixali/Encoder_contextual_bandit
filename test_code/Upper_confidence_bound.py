
### so the question is why I do not just 
### calculate the total reward in each column and then just pick the column with largest value?
import pandas as pd
import random
import math

random.seed(1000)
df = pd.read_csv("Ads_Optimisation.csv")

def upper_confidence_bound(df):

    rowNum = 10000
    colNum = 10

    numbers_of_selections = [0] * colNum
    sums_of_reward = [0] * colNum
    values = df.values
    total_reward = 0
    ads_selected = []

    ### so one round mean that go through all the arms once?
    for n in range(0, rowNum):
        max_upper_bound = 0

        for i in range(0, colNum):

            if (numbers_of_selections[i] > 0):
                average_reward = sums_of_reward[i] / numbers_of_selections[i]

                delta_i = math.sqrt(2* math.log(n+1)/numbers_of_selections[i])

                upper_bound = average_reward + delta_i

            else:
                upper_bound = 1e400

            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                ad = i

        ads_selected.append(ad)
        numbers_of_selections[ad] += 1
        reward = values[n, ad]
        sums_of_reward[ad] += reward
        total_reward += reward

    return total_reward


print(upper_confidence_bound(df))
