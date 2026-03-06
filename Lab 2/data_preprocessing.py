import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch

def read_data():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    return train_df,test_df

def data_clean(df):
    df = df.dropna()
    return df

def data_split(df):

    train_df, test_df  = read_data()

    train_df = data_clean(train_df)
    test_df = data_clean(test_df)

    X= train_df.drop(['highUptake_mol'],axis=1)
    y= train_df['highUptake_mol']

    X_train,X_validation,y_train,y_validation = train_test_split(X,y,test_size=.3,random_state=42)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_validation_tensor = torch.tensor(X_validation.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_validation_tensor = torch.tensor(y_validation.values, dtype=torch.float32)

    test_df_tensor = torch.tensor(test_df.values, dtype=torch.float32)


    return (X_train_tensor, X_validation_tensor, y_train_tensor, y_validation_tensor), (test_df_tensor)