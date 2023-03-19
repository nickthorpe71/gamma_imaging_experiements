from typing import Tuple
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler


def naive_bayes(df_X_train: DataFrame, df_y_train: DataFrame, df_X_test: DataFrame) -> DataFrame:
    nb_model = GaussianNB()
    nb_model.fit(df_X_train, df_y_train)
    df_y_pred = nb_model.predict(df_X_test)
    
    return df_y_pred