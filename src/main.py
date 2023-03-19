from typing import Tuple
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

def import_data() -> DataFrame:
    cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
    df = pd.read_csv("src/data/magic04.data", names=cols)
    return df

def format_class_to_int(df: DataFrame) -> DataFrame:
    df["class"] = (df["class"] == "g").astype(int) # convert class to int: g = 1, h = 0
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
    [df_train, X_train, y_train] = scale_data(df_train, oversample=True)
    [df_valid, X_valid, y_valid] = scale_data(df_valid)
    [df_test, X_test, y_test] = scale_data(df_test)
    
    print(len(y_train))
    print(sum(y_train == 1))
    print(sum(y_train == 0))


if __name__ == "__main__":
    main()