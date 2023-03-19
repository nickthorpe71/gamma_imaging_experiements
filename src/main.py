from typing import List, Tuple
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

from src.plotting import plot_data

def import_data() -> Tuple[DataFrame, List[str]]:
    cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
    df = pd.read_csv("src/data/magic04.data", names=cols)
    return [df, cols]

def format_class_to_int(df: DataFrame) -> DataFrame:
    df["class"] = df["class"].map({"g": 0, "h": 1})
    return df
        

def main():
    [data, cols] = import_data()
    numeric_data = format_class_to_int(data)
    plot_data(numeric_data, cols)


if __name__ == "__main__":
    main()