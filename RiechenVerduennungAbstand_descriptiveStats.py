#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 19:09:44 2022

@author: joern
"""

# %% imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from scipy.stats import chi2_contingency
from scipy import stats

from RiechenVerduennungAbstand_readandexploredata import FinalDataSetPreprocessed
from RiechenVerduennungAbstand_readandexploredata import dfRiechenVerduennungAbstand

# %% Data

Normosmia_vs_hyposmia_Data = FinalDataSetPreprocessed.copy()
FinalDataSetPreprocessed.columns
Normosmia_vs_hyposmia_Data["Olf. diagnosis"] = dfRiechenVerduennungAbstand["An0_Hyp1_Norm2"]

# %% Analyze differnces Normosmia verus hyposmia

p_OlfDiag = []
for i in range(Normosmia_vs_hyposmia_Data.shape[1]-1):
    print(Normosmia_vs_hyposmia_Data.columns[i])
    Statistic = stats.mannwhitneyu(*[group[Normosmia_vs_hyposmia_Data.columns[i]].values
                                     for name, group in Normosmia_vs_hyposmia_Data.groupby(Normosmia_vs_hyposmia_Data.iloc[:, -1])])
    p_OlfDiag.append(Statistic.pvalue)
    print(Statistic)

dfp_OlfDiag = pd.DataFrame(p_OlfDiag, columns=[
                           "p_value"], index=Normosmia_vs_hyposmia_Data.columns.tolist()[:-1])

fig, ax = plt.subplots(figsize=(18, 16))
ax = sns.barplot(x=dfp_OlfDiag.index, y=-np.log10(dfp_OlfDiag["p_value"]))
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_ylabel("-log10(p)")
ax.axhline(-np.log10(0.05), color="salmon", linestyle="dotted")
ax.axhline(-np.log10(0.05 /
           (Normosmia_vs_hyposmia_Data.shape[1]-1)), color="blue", linestyle="dotted")

contigency = pd.crosstab(
    dfRiechenVerduennungAbstand['sex_0f'], Normosmia_vs_hyposmia_Data['Olf. diagnosis'])
contigency
c, p, dof, expected = chi2_contingency(contigency)

