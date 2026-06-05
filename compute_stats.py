import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

if __name__=='__main__':
    ga = pd.read_csv("ga_results.csv")
    de = pd.read_csv("de_results.csv")

    de_summary = de.groupby("initialization")["test_fitness"].mean()
    best_de_init = de_summary.idxmax()

    best_de = de[de["initialization"] == best_de_init]

    ga_scores = ga["test_fitness"]
    de_scores = best_de["test_fitness"]

    stat, p_value = mannwhitneyu(ga_scores, de_scores, alternative="two-sided")

    print("Mann-Whitney U:", stat)
    print("p-value:", p_value)