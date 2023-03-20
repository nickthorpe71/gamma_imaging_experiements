from typing import Tuple
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler

# models
from models.knn import knn
from models.naive_bayes import naive_bayes
from models.svm import run_svm

def import_data() -> DataFrame:
    cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
    df = pd.read_csv("src/data/magic04.data", names=cols)
    return df

def format_class_to_int(df: DataFrame) -> DataFrame:
    df["class"] = (df["class"] == "g").astype(int) # convert class to int: g = 1, h = 0
    # df["class"] = df["class"].apply(lambda x: 1 if x == 1 else -1) # convert class to int: g = 1, h = -1
    return df

def split_data(df: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
    train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])
    return [train, valid, test]

def scale_data(df: DataFrame, oversample: bool=False) -> Tuple[DataFrame, DataFrame, DataFrame]:
    X = df[df.columns[:-1]].values # 2D array of features (excluding class)
    y = df[df.columns[-1]].values # 1D array of class labels
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X) # scale features
    
    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)
    
    # combine features and class labels back together after scaling
    data = np.hstack((X, np.reshape(y, (-1, 1)))) # hstack = horizontal stack, -1 = unknown number of rows (same as len(y))
    
    return data, X , y
    

def main():
    df_data = import_data()
    
    # format data
    df_numeric_data = format_class_to_int(df_data)
    # split data
    [df_train, df_valid, df_test] = split_data(df_numeric_data)
    # scale data / preprocess data
    [df_train, df_X_train, df_y_train] = scale_data(df_train, oversample=True)
    [df_valid, df_X_valid, df_y_valid] = scale_data(df_valid)
    [df_test, df_X_test, df_y_test] = scale_data(df_test)
    
    run_svm(df_X_train, df_y_train, df_X_test, df_y_test)


if __name__ == "__main__":
    main()