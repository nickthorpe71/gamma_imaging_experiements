from typing import List
from pandas import DataFrame
import matplotlib.pyplot as plt

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