import pandas as pd


#logic to get only the required columns from original survey file and save it (run only once)
'''
df = pd.read_csv("surveydata/survey.csv")
newdf = df.loc[:,['CASEID', 'YYYYMM', 'YYYY', 'ID', 'PAGO', 'HOM', 'GOVT', 'AGE', 'SEX', 'MARRY', 'REGION', 'EDUC']]
newdf.to_csv('surveydata/survey_subset.csv', index=False)

exit()
'''




Q1code = "GOVT" #"As the economic policy of the government ‐‐ I mean steps taken to fight inflation or unemployment ‐‐ would you say the government is doing a good job, only fair, or a poor job?
Q2code = "HOM" #"Generally speaking, do you think now is a good time or bad time to buy a house?"
Q3code =  "PAGO" #"We are interested in how people are getting along financially these days. Would you say that you (and your family living there) are better or worse off financially than you were ayear ago?"

keywords  ={}



Q1dict = {
    1: 5.0,
    3: 4.0,
    5: 3.0,
    8: 1.0,
    9: 0.0
}

Q2dict = {
    1: 5.0,
    3: 4.0,
    5: 3.0,
    8: 1.0,
    9: 0.0
}

Q3dict = {
    1: 5.0,
    3: 4.0,
    5: 3.0,
    8: 1.0,
    9: 0.0
}

ALL_DEMOGRAPHICS = ['AGE', 'SEX', 'MARRY', 'REGION', 'EDUC']

def get_name(demographics):
    fname = "surveydata/aggregated"
    for val in demographics:
        fname+="_"
        fname+=val
    return fname+".csv"


def generate_data(demographics = [], start="2020-01-01", end="2022-05-31"):
    df = pd.read_csv("surveydata/survey_subset.csv")
    # Filter the DataFrame based on the date range
    df = df[df['YYYYMM'] >= int(start[:4]+start[5:7])]
    df = df[df['YYYYMM'] <= int(end[:4]+end[5:7])]
    
    #transform Q1
    df['GOVT'] = df['GOVT'].apply(lambda x: Q1dict[x])
    df['HOM'] = df['HOM'].apply(lambda x: Q2dict[x])
    df['PAGO'] = df['PAGO'].apply(lambda x: Q3dict[x])

    print(demographics)
    groub_by_cols = ['YYYYMM']+demographics
    df_agg = df.groupby(groub_by_cols)[['GOVT', 'HOM', 'PAGO']].mean()
    
    df_agg = df_agg.reset_index()

    #TODO: make this more generic
    df_agg['GOVT'] = df_agg['GOVT']/4.223021582733813
    df_agg['HOM'] = df_agg['HOM']/4.45646437994723
    df_agg['PAGO'] = df_agg['PAGO']/4.5075
    
    return df_agg.astype(str)
    #df_agg.to_csv(get_name(demographics))


#generate_data(['SEX', 'MARRY'])
