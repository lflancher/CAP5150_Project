import pydp as dp
from pydp.algorithms.laplacian import (
    BoundedSum,
    BoundedMean,
    BoundedStandardDeviation,
    Count,
    Max,
    Min,
    Median,
    Percentile,
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

def twenty_fifth_percentile_fare() -> float:
    return df["Fare"].quantile(0.25)

def seventy_fifth_percentile_fare() -> float:
    return df["Fare"].quantile(0.75)

def min_fare() -> float:
    return df["Fare"].min()

def max_fare() -> float:
    return df["Fare"].max()

def private_mean_fare(privacy_budget: float) -> float:
    x = BoundedMean(privacy_budget, lower_bound = 0, upper_bound = 600, dtype = "float")
    return x.quick_result(list(df["Fare"]))

def private_median_fare(privacy_budget: float) -> float:
    x = Median(privacy_budget, lower_bound = 0, upper_bound  = 600, dtype = "float")
    return x.quick_result(list(df["Fare"]))

def private_std_fare(privacy_budget: float) -> float:
    x = BoundedStandardDeviation(privacy_budget, lower_bound = 0, upper_bound = 600, dtype = "float")
    return x.quick_result(list(df["Fare"]))

def private_twenty_fifth_percentile_fare(privacy_budget: float) -> float:
    x = Percentile(privacy_budget, lower_bound = 0, upper_bound = 600, dtype = "float", percentile = 0.25)
    return x.quick_result(list(df["Fare"]))

def private_seventy_fifth_percentile_fare(privacy_budget: float) -> float:
    x = Percentile(privacy_budget, lower_bound = 0, upper_bound = 600, dtype = "float", percentile = 0.75)
    return x.quick_result(list(df["Fare"]))

def private_min_fare(privacy_budget: float) -> float:
    x = Min(privacy_budget, lower_bound = 0, upper_bound = 600, dtype = "float")
    return x.quick_result(list(df["Fare"]))

def private_max_fare(privacy_budget: float) -> float:
    x = Max(privacy_budget, lower_bound = 0, upper_bound = 600, dtype = "float")
    return x.quick_result(list(df["Fare"]))

#Here I run each several times, just to gauge accuracy
privatemeantotal         = float(0)
privatemediantotal       = float(0)
privatestdtotal          = float(0)
privatetwentyfifthtotal  = float(0)
privateseventyfifthtotal = float(0)
privatemintotal          = float(0)
privatemaxtotal          = float(0)
totalruns = 100
for i in range(1, totalruns):
    privatemeantotal         = privatemeantotal + private_mean_fare(0.8)
    privatemediantotal       = privatemediantotal + private_median_fare(0.8)
    privatestdtotal          = privatestdtotal + private_std_fare(0.8)
    privatetwentyfifthtotal  = privatetwentyfifthtotal + private_twenty_fifth_percentile_fare(0.8)
    privateseventyfifthtotal = privateseventyfifthtotal + private_seventy_fifth_percentile_fare(0.8)
    privatemintotal          = privatemintotal + private_min_fare(0.8)
    privatemaxtotal          = privatemaxtotal + private_max_fare(0.8)

avgmean         = float(privatemeantotal   / totalruns)
avgmedian       = float(privatemediantotal / totalruns)
avgstd          = float(privatestdtotal    / totalruns)
avgtwentyfifth  = float(privatetwentyfifthtotal / totalruns)
avgseventyfifth = float(privateseventyfifthtotal / totalruns)
avgmintotal     = float(privatemintotal / totalruns)
avgmaxtotal     = float(privatemaxtotal / totalruns)



privmean = private_mean_fare(0.8)
privmedian = private_median_fare(0.8)
privstd = private_std_fare(0.8)
privtwentyfifth = private_twenty_fifth_percentile_fare(0.8)
privseventyfifth = private_seventy_fifth_percentile_fare(0.8)
privmin = private_min_fare(0.8)
privmax = private_max_fare(0.8)       


with open("Results/Traditional_Implementation.csv", "w") as f:
    f.write("Statistic,Non-Private,Private_Traditional,Average_Private_Traditional\n")
    f.write(f"Mean,{mean_fare()},{privmean},{avgmean}\n")
    f.write(f"Median,{median_fare()},{privmedian},{avgmedian}\n")
    f.write(f"STD,{std_fare()},{privstd},{avgstd}\n")
    f.write(f"25th Percentile,{twenty_fifth_percentile_fare()},{privtwentyfifth},{avgtwentyfifth}\n")
    f.write(f"75th Percentile,{seventy_fifth_percentile_fare()},{privseventyfifth},{avgseventyfifth}\n")
    f.write(f"Minimum,{min_fare()},{privmin},{avgmintotal}\n")
    f.write(f"Maximum,{max_fare()},{privmax},{avgmaxtotal}\n")