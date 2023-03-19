import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

def import_data() -> DataFrame:
    cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
    df = pd.read_csv("src/data/magic04.data", names=cols)
    return df

def format_class_to_int(df: DataFrame) -> DataFrame:
    df["class"] = df["class"].map({"g": 0, "h": 1})
    return df

def main():
    data = format_class_to_int(import_data())
    print(data.head())


if __name__ == "__main__":
    main()