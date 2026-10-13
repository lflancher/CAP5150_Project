import pydp as dp
from pydp.algorithms.laplacian import (
    BoundedSum,
    BoundedMean,
    BoundedStandardDeviation,
    Count,
    Max,
    Min,
    Median,
)
import pandas as pd
import statistics 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Datasets/titanic_clean.csv')

#Pure interpretations of the raw data
def mean_fare() -> float:
    return df["Fare"].mean()

def median_fare() -> float:
    return df["Fare"].median()

def std_fare() -> float:
    return df["Fare"].std()

def private_mean_fare(privacy_budget: float) -> float:
    x = BoundedMean(privacy_budget, lower_bound = 0, upper_bound = 600, dtype = "float")
    return x.quick_result(list(df["Fare"]))

def private_median_fare(privacy_budget: float) -> float:
    x = Median(privacy_budget, lower_bound = 0, upper_bound  = 600, dtype = "float")
    return x.quick_result(list(df["Fare"]))

def private_std_fare(privacy_budget: float) -> float:
    x = BoundedStandardDeviation(privacy_budget, lower_bound = 0, upper_bound = 600, dtype = "float")
    return x.quick_result(list(df["Fare"]))



#Here I run each several times, just to gauge accuracy
privatemeantotal = float(0)
privatemediantotal = float(0)
privatestdtotal = float(0)
totalruns = 100
for i in range(1, totalruns):
    privatemeantotal   = privatemeantotal + private_mean_fare(0.8)
    privatemediantotal = privatemediantotal + private_median_fare(0.8)
    privatestdtotal    = privatestdtotal + private_std_fare(0.8)

avgmean   = float(privatemeantotal   / totalruns)
avgmedian = float(privatemediantotal / totalruns)
avgstd    = float(privatestdtotal    / totalruns)


print("Mean: ", mean_fare())
print("Median: ", median_fare())
print("STD: ", std_fare())

print("Private Mean: ", private_mean_fare(0.8))
print("Private Median: ", private_median_fare(0.8))
print("Private STD: ", private_std_fare(0.8))

print("Averge Private Mean over 100 Tries: ", str(avgmean))
print("Average Private Median over 100 tries: ", str(avgmedian))
print("Average Private STD over 100 tries: ", str(avgstd))

