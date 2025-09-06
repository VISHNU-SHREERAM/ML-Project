import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(dfPath1, dfPath2):
    df = pd.read_csv(dfPath1)
    df2 = pd.read_csv(dfPath2)
    
    df = pd.concat((df, df2))
    return df

def handle_bp(df):
    lowRow = []
    highRow = []


    for val in df['Blood Pressure']:
        sp = val.split('/')

        highRow.append(int(sp[0]))
        lowRow.append(int(sp[1]))

    df['Diastolic Pressure'] = np.array(lowRow)
    df['Systolic Pressure'] = np.array(highRow)
    df = df.drop('Blood Pressure', axis = 1)
    return df

def categorical_features(df):
    return [key for key in df if (df[key].dtype == object)]

def numeric_features(df):
    return [key for key in df if (df[key].dtype != object and key != 'Stress Level')]

def encode(df):
    catData = categorical_features(df)

    label_encoding_dict = {}
    for category in catData:
        labelEncoder = LabelEncoder()
        labelEncoder.fit(df[category])
        df[category] = labelEncoder.transform(df[category])
        label_encoding_dict[category] = labelEncoder
    
    return df, label_encoding_dict

def standardise(df):
    numData = numeric_features(df)
    for f in numData:
        df[f] = StandardScaler().fit_transform(df[[f]])
    
    return df