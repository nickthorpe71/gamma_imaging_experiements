from typing import List, Tuple
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

def import_data() -> Tuple[DataFrame, List[str]]:
    cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
    df = pd.read_csv("src/data/magic04.data", names=cols)
    return [df, cols]

def format_class_to_int(df: DataFrame) -> DataFrame:
    df["class"] = df["class"].map({"g": 0, "h": 1})
    return df

def plot_data(df: DataFrame, cols: List[str]) -> None:
    for label in cols[:-1]:
        plt.hist(df[df["class"]==1][label], color='blue', label="gamma", alpha=0.7, density=True)
        plt.hist(df[df["class"]==0][label], color='red', label="hadron", alpha=0.7, density=True)
        plt.title(label)
        plt.ylabel("Probability")
        plt.xlabel(label)
        plt.legend()
        plt.savefig(f"src/plots/{label}.png")
        plt.close()
        
        
        

def main():
    [data, cols] = import_data()
    numeric_data = format_class_to_int(data)
    plot_data(numeric_data, cols)


if __name__ == "__main__":
    main()