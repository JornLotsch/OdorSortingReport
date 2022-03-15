#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 16:14:53 2022

@author: joern
"""

from scipy.stats import pearsonr, spearmanr
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from compute_PCA import perform_pca
from ABCanalysis import ABC_analysis

from RiechenVerduennungAbstand_readandexploredata import FinalDataSetPreprocessed
from RiechenVerduennungAbstand_readandexploredata import dfRiechenVerduennungAbstand
from RiechenVerduennungAbstand_readandexploredata import groups_of_analyzed_variables


# %% Funtions
def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = pearsonr(df[r], df[c])[1]
    return pvalues

# %% Data

DataForCorrelationPCA = FinalDataSetPreprocessed.copy()

# %% Correlation matrix


CorrelationsP = calculate_pvalues(DataForCorrelationPCA)
CorrelationsP.shape
CorrelationsP.replace(0, np.NaN, inplace=True)

CorrelationsR = DataForCorrelationPCA.corr(method="pearson")
CorrelationsR.replace(1, np.NaN, inplace=True)

maxCorrR = np.nanmax(CorrelationsR.abs())
maxCorrR**2
BlueGreen = sns.color_palette(
    "blend:darkblue,dodgerblue,whitesmoke,chartreuse,green", as_cmap=True)
fig, ax = plt.subplots(figsize=(18, 16))
sns.heatmap(CorrelationsR, annot=True, annot_kws={
            "color": "black", "va": "bottom"}, cmap=BlueGreen, vmin=-maxCorrR, vmax=maxCorrR)  # was "BrBG"
#for t in res.texts: t.set_text(t.get_text() + " %")
sns.heatmap(CorrelationsP, mask=CorrelationsP >= 0.05, annot=True, annot_kws={
            "color": "red", "va": "top"}, cmap="Greys_r", alpha=0, cbar=False)  # was "BrBG"

ax.add_patch(patches.Rectangle((2, 2), 15, 15,
             edgecolor='salmon', fill=False, lw=2))

len(groups_of_analyzed_variables.values())  # From preprocessing
rectangel_sizes = [len(x) for x in groups_of_analyzed_variables.values()]
start = [0, 0]
for i in rectangel_sizes[:6]:
    ax.add_patch(patches.Rectangle(
        (start),
        i,
        i,
        edgecolor='black',
        fill=False,
        lw=2
    ))
    start = [sum(x) for x in zip(start, [i, i])]

np.nanmax(CorrelationsR**2)


# %% PCA

PCA_data = DataForCorrelationPCA.copy()
PCA_data.drop(['Age', 'BMI^-2'], axis = 1, inplace = True)
PCA_data = pd.DataFrame(StandardScaler().fit_transform(
    PCA_data), columns=PCA_data .columns)

y2 = dfRiechenVerduennungAbstand["An0_Hyp1_Norm2"]

PCA_olfactory, PCA_olfactory_feature_importance = perform_pca(
    PCA_data, target=y2, PC_criterion="KaiserGuttman", plotReduced=1)
PCA_features = ABC_analysis(PCA_olfactory_feature_importance)
