#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 16:14:53 2022

@author: joern
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import miceforest as mf
import matplotlib.gridspec as gridspec

from box_and_heatplots import box_and_heatplot
from explore_tukey_lop import explore_tukey_lop

# %% Thomas alternative counting
alternativeOderCounting = False

# %% Read data
pfad_o = "/home/joern/Aktuell/RiechenVerduennungAbstand/"
pfad_u1 = "09Originale/"
filename = "data_anne_huster_dec_2021.xlsx"
sheetname = "data"

dfRiechenVerduennungAbstand = pd.read_excel(
    pfad_o + pfad_u1 + filename, sheet_name=sheetname)
dfRiechenVerduennungAbstand.columns

# %% Correct YOB <- 1900
dfRiechenVerduennungAbstand.iloc[dfRiechenVerduennungAbstand["YOB"].values < 1900]
dfRiechenVerduennungAbstand["YOB"] = dfRiechenVerduennungAbstand["YOB"].replace([
                                                                                1191], 1991)
dfRiechenVerduennungAbstand["YOB"] = dfRiechenVerduennungAbstand["YOB"].replace([
                                                                                1194], 1994)
dfRiechenVerduennungAbstand["Age"] = 2020 - dfRiechenVerduennungAbstand["YOB"]
dfRiechenVerduennungAbstand.shape
dfRiechenVerduennungAbstand.shape

# %% Basic descriptive statistics
dfRiechenVerduennungAbstand.Age.describe()
dfRiechenVerduennungAbstand.sex_0f.value_counts()
dfRiechenVerduennungAbstand.BMI.describe()
dfRiechenVerduennungAbstand.An0_Hyp1_Norm2.value_counts()

# %% Visual exploration of all variables
groups_of_variables = {
    "Demographics": ["Age", "sex_0f", "KG in kg", "Height in cm", "BMI"],
    "Importance_of_Olfaction": ["ImportofO_Lying", "ImportofO_Total", "ImportofO_Evaluation", "ImportofO_Application", "ImportofO_Consequence"],
    "Olfactory_Subtests": ["Thr", "Dis", "Id", "TDI"],
    "Peanunt_butter_test": ["Distance right nostril", "Distance left nostril", "Dist Difference left minus right", "Dist Difference abs left right"],
    "Odor_sorting_task_PEA": ["Dilution series PEA rank 1", "Dilution series PEA rank 2", "Dilution series PEA rank 3", "Dilution series PEA rank 4",
                              "Dilution series PEA rank 5", "Dilution series PEA time required in s", "PEA Dilution series score", "PEA success by time times 1000"],
    "Odor_sorting_task_EUG": ["Dilution series eucenol rank 1", "Dilution series eucenol rank 2", "Dilution series eucenol rank 3",
                              "Dilution series eucenol rank 4", "Dilution series eucenol rank 5", "euc Dilution series time", "euc Dilution series score",
                              "Euc success by time times 1000"],
    "Odor_sorting_task_combd": ["rank score M Pea plus Euc", "rank success M Pea plus Euc"],
    "Lateralisation_task": ["Lat correct assignment right nostril", "Lat correct assignment left nostril", "Lat correct assignments overall",
                            "Lat left minus right", "Lat absolute difference"],
    "Discrimination_task": ["(-) - Limonene / (+) - Limonene", "(-) - Carvone / (+) - Carvone", "(-) - Fenchone / (+) - Fenchone",
                            "(-) - 2 butanol / (+) - 2 butanol", "enantiomer sum"],
    "Threshold_after_exposition": ["PEA threshold after PEA clip", "PEA clip minus PEA standard"]}

# for i, variableGroups in enumerate(list(groups_of_variables.keys())):
#     variables = groups_of_variables[variableGroups]
#     data_subset = copy.copy(dfRiechenVerduennungAbstand[list(variables)])
#     box_and_heatplot(data=data_subset, title=variableGroups,
#                      scale=True, cmap="viridis")


# %% Visual exploration of non-generated variables
groups_of_nongenerated_variables = {
    "Demographics": ["Age", "sex_0f", "KG in kg", "Height in cm", "BMI"],
    "Importance_of_Olfaction": ["ImportofO_Evaluation", "ImportofO_Application", "ImportofO_Consequence"],
    "Olfactory_Subtests": ["Thr", "Dis", "Id"],
    "Peanunt_butter_test": ["Distance right nostril", "Distance left nostril"],
    "Odor_sorting_task_PEA": ["Dilution series PEA rank 1", "Dilution series PEA rank 2", "Dilution series PEA rank 3", "Dilution series PEA rank 4",
                              "Dilution series PEA rank 5", "Dilution series PEA time required in s", "PEA Dilution series score"],
    "Odor_sorting_task_EUG": ["Dilution series eucenol rank 1", "Dilution series eucenol rank 2", "Dilution series eucenol rank 3",
                              "Dilution series eucenol rank 4", "Dilution series eucenol rank 5", "euc Dilution series time", "euc Dilution series score"],
    "Odor_sorting_task_combd": [],
    "Lateralisation_task": ["Lat correct assignment right nostril", "Lat correct assignment left nostril"],
    "Discrimination_task": ["(-) - Limonene / (+) - Limonene", "(-) - Carvone / (+) - Carvone", "(-) - Fenchone / (+) - Fenchone",
                            "(-) - 2 butanol / (+) - 2 butanol"],
    "Threshold_after_exposition": ["PEA threshold after PEA clip"]}

all_nongenerated_variables = [item for sublist in [
    a for a in groups_of_nongenerated_variables.values()] for item in sublist]

# %% Transformations

# for i, variable in enumerate(all_nongenerated_variables):
#     data_subset = copy.copy(dfRiechenVerduennungAbstand[variable])
#     explore_tukey_lop(data=data_subset)
#     # explore_tukey_lop(data=data_subset, outlierremoval = True)

dfRiechenVerduennungAbstand["BMI^-2"] = np.power(
    dfRiechenVerduennungAbstand["BMI"].astype("float"), -2)
dfRiechenVerduennungAbstand["log Distance right nostril"] = np.log(
    dfRiechenVerduennungAbstand["Distance right nostril"].astype("float"))
dfRiechenVerduennungAbstand["log Distance left nostril"] = np.log(
    dfRiechenVerduennungAbstand["Distance left nostril"].astype("float"))
dfRiechenVerduennungAbstand["log PEA threshold after PEA clip"] = np.log(
    dfRiechenVerduennungAbstand["PEA threshold after PEA clip"].astype("float"))
dfRiechenVerduennungAbstand["log Thr"] = np.log(
    dfRiechenVerduennungAbstand["Thr"].astype("float"))
# %% Outlier removal
groups_of_nongenerated_variables_transformed = {
    "Demographics": ["Age", "sex_0f", "KG in kg", "Height in cm", "BMI^-2"],
    "Importance_of_Olfaction": ["ImportofO_Evaluation", "ImportofO_Application", "ImportofO_Consequence"],
    "Olfactory_Subtests": ["log Thr", "Dis", "Id"],
    "Peanunt_butter_test": ["log Distance right nostril", "log Distance left nostril"],
    "Odor_sorting_task_PEA": ["Dilution series PEA rank 1", "Dilution series PEA rank 2", "Dilution series PEA rank 3", "Dilution series PEA rank 4",
                              "Dilution series PEA rank 5", "Dilution series PEA time required in s"],
    "Odor_sorting_task_EUG": ["Dilution series eucenol rank 1", "Dilution series eucenol rank 2", "Dilution series eucenol rank 3",
                              "Dilution series eucenol rank 4", "Dilution series eucenol rank 5", "euc Dilution series time"],
    "Odor_sorting_task_combd": [],
    "Lateralisation_task": ["Lat correct assignment right nostril", "Lat correct assignment left nostril"],
    "Discrimination_task": ["(-) - Limonene / (+) - Limonene", "(-) - Carvone / (+) - Carvone", "(-) - Fenchone / (+) - Fenchone",
                            "(-) - 2 butanol / (+) - 2 butanol"],
    "Threshold_after_exposition": ["log PEA threshold after PEA clip"]}

all_nongenerated_variables_transformed = [item for sublist in [
    a for a in groups_of_nongenerated_variables_transformed.values()] for item in sublist]
all_nongenerated_olfactory_variables_transformed = all_nongenerated_variables_transformed[len(
    groups_of_nongenerated_variables_transformed["Demographics"]):len(all_nongenerated_variables_transformed)]


dfRiechenVerduennungAbstand_o = copy.copy(
    dfRiechenVerduennungAbstand[all_nongenerated_variables_transformed])
dfRiechenVerduennungAbstand_o.isna().sum()

dfRiechenVerduennungAbstand_o = dfRiechenVerduennungAbstand_o.mask(dfRiechenVerduennungAbstand_o.sub(
    dfRiechenVerduennungAbstand_o.mean()).div(dfRiechenVerduennungAbstand_o.std()).abs().gt(1113))
# dfRiechenVerduennungAbstand_o["Age"] = dfRiechenVerduennungAbstand["Age"]
# dfRiechenVerduennungAbstand_o["BMI"] = dfRiechenVerduennungAbstand["BMI"]
# dfRiechenVerduennungAbstand_o["KG in kg"] = dfRiechenVerduennungAbstand["KG in kg"]
dfRiechenVerduennungAbstand_o.isna().sum()

PercentNA = dfRiechenVerduennungAbstand_o.isna().sum() / dfRiechenVerduennungAbstand_o.shape[0] * 100
fig = plt.figure(figsize=(20, 10))
ax = sns.barplot(x=PercentNA.index, y=PercentNA,  color="dodgerblue")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.axhline(20, linewidth=4, color="salmon")

# %% Imputation and visual exploration of olfactory variables
data_subset = copy.copy(
    dfRiechenVerduennungAbstand_o[all_nongenerated_olfactory_variables_transformed])

# Create kernel
kds = mf.ImputationKernel(
    data_subset,
    datasets=1,
    save_all_iterations=True,
    random_state=42
)
# Impute with 100 iterations
kds.mice(100)
print(kds)

dfRiechenVerduennungAbstand_imputed = kds.complete_data(
    dataset=0, inplace=False)
fig = plt.figure(figsize=(40, 20))
kds.plot_feature_importance(dataset=0, annot=True,
                            cmap="YlGnBu", vmin=0, vmax=1)

# %% (Re)Create composed scores

def f(x): return 0 if x != i1 else 1

variables_Sorting_PEA = groups_of_nongenerated_variables["Odor_sorting_task_PEA"][0:5]
dfvariables_Sorting_PEA = copy.copy(
    dfRiechenVerduennungAbstand_imputed[variables_Sorting_PEA])
variables_Sorting_EUG = groups_of_nongenerated_variables["Odor_sorting_task_EUG"][0:5]
dfvariables_Sorting_EUG = copy.copy(
    dfRiechenVerduennungAbstand_imputed[variables_Sorting_EUG])

for i, variable in enumerate(dfvariables_Sorting_PEA):
    i1 = i + 1
    dfvariables_Sorting_PEA[variable] = dfvariables_Sorting_PEA[variable].map(f)

for i, variable in enumerate(dfvariables_Sorting_EUG):
    i1 = i + 1
    dfvariables_Sorting_EUG[variable] = dfvariables_Sorting_EUG[variable].map(f)

if alternativeOderCounting:
    errordistances_PEA = dfvariables_Sorting_PEA.sum(axis=1)
    errordistances_EUG = dfvariables_Sorting_EUG.sum(axis=1)

else:
    errordistances_PEA = abs(dfRiechenVerduennungAbstand_imputed[variables_Sorting_PEA] - [1, 2, 3, 4, 5]).sum(axis=1)
    errordistances_EUG = abs(dfRiechenVerduennungAbstand_imputed[variables_Sorting_EUG] - [1, 2, 3, 4, 5]).sum(axis=1)
    
    
variables_Discrimination_task = groups_of_nongenerated_variables["Discrimination_task"]
correct_Discrimination_task = dfRiechenVerduennungAbstand_imputed[variables_Discrimination_task].sum(
    axis=1)

# %% Create data set for analyses
groups_of_analyzed_variables = {
    "Demographics": ["Age", "BMI^-2"],
    "Importance_of_Olfaction": ["ImportofO_Evaluation", "ImportofO_Application", "ImportofO_Consequence"],
    "Olfactory_Subtests": ["log Thr", "Dis", "Id"],
    "Peanunt_butter_test": ["log Distance right nostril", "log Distance left nostril"],
    "Odor_sorting_task_PEA": ["errordistances_PEA", "errordistances_PEA_timecorrected"],
    "Odor_sorting_task_EUG": ["errordistances_EUG", "errordistances_EUG_timecorrected"],
    "Lateralisation_task": ["Lat correct assignments overall"],
    "Discrimination_task": ["correct_Discrimination_task"],
    "Threshold_after_exposition": ["log PEA threshold after PEA clip"]}

analyzed_variables = [item for sublist in [
    a for a in groups_of_analyzed_variables.values()] for item in sublist]

dfRiechenVerduennungAbstand_imputed_analyzed = copy.copy(
    dfRiechenVerduennungAbstand_imputed)
dfRiechenVerduennungAbstand_imputed_analyzed["errordistances_PEA"] = errordistances_PEA
dfRiechenVerduennungAbstand_imputed_analyzed["errordistances_PEA_timecorrected"] = dfRiechenVerduennungAbstand_imputed_analyzed[
    "errordistances_PEA"] / dfRiechenVerduennungAbstand_imputed_analyzed["Dilution series PEA time required in s"]
dfRiechenVerduennungAbstand_imputed_analyzed["errordistances_EUG"] = errordistances_EUG
dfRiechenVerduennungAbstand_imputed_analyzed["errordistances_EUG_timecorrected"] = dfRiechenVerduennungAbstand_imputed_analyzed[
    "errordistances_EUG"] / dfRiechenVerduennungAbstand_imputed_analyzed["euc Dilution series time"]
dfRiechenVerduennungAbstand_imputed_analyzed["correct_Discrimination_task"] = correct_Discrimination_task
dfRiechenVerduennungAbstand_imputed_analyzed[[
    "Age", "BMI^-2"]] = dfRiechenVerduennungAbstand[["Age", "BMI^-2"]]
dfRiechenVerduennungAbstand_imputed_analyzed["Lat correct assignments overall"] = dfRiechenVerduennungAbstand_imputed_analyzed[
    "Lat correct assignment right nostril"] + dfRiechenVerduennungAbstand_imputed_analyzed["Lat correct assignment left nostril"]
dfRiechenVerduennungAbstand_imputed_analyzed = dfRiechenVerduennungAbstand_imputed_analyzed[
    analyzed_variables]

# for i, variableGroups in enumerate(list(groups_of_analyzed_variables.keys())):
#     variables = groups_of_analyzed_variables[variableGroups]
#     data_subset = copy.copy(
#         dfRiechenVerduennungAbstand_imputed_analyzed[list(variables)])
#     box_and_heatplot(data=data_subset, title=variableGroups,
#                      scale=True, cmap="viridis")


groups_of_analyzed_olfactory_variables = {
    "Importance_of_Olfaction": ["ImportofO_Evaluation", "ImportofO_Application", "ImportofO_Consequence"],
    "Olfactory_Subtests": ["log Thr", "Dis", "Id"],
    "Peanunt_butter_test": ["log Distance right nostril", "log Distance left nostril"],
    "Odor_sorting_task_PEA": ["errordistances_PEA", "errordistances_PEA_timecorrected"],
    "Odor_sorting_task_EUG": ["errordistances_EUG", "errordistances_EUG_timecorrected"],
    "Lateralisation_task": ["Lat correct assignments overall"],
    "Discrimination_task": ["correct_Discrimination_task"],
    "Threshold_after_exposition": ["log PEA threshold after PEA clip"]}

analyzed_olfactory_variables = [item for sublist in [
    a for a in groups_of_analyzed_olfactory_variables.values()] for item in sublist]

dfRiechenVerduennungAbstand_olfactory_imputed_analyzed = copy.copy(
    dfRiechenVerduennungAbstand_imputed_analyzed[analyzed_olfactory_variables])
dfRiechenVerduennungAbstand_olfactory_imputed_analyzed.shape
dfRiechenVerduennungAbstand_olfactory_imputed_analyzed.columns

# for i, variable in enumerate(analyzed_olfactory_variables):
#     data_subset = copy.copy(
#         dfRiechenVerduennungAbstand_olfactory_imputed_analyzed[variable])
#     explore_tukey_lop(data=data_subset)


# %% Plot raw data

groups_of_raw_values_of_analyzed_variables = {
    "Importance_of_Olfaction": ["ImportofO_Evaluation", "ImportofO_Application", "ImportofO_Consequence"],
    "Olfactory_Subtests": ["Thr", "Dis", "Id"],
    "Peanunt_butter_test": ["Distance right nostril", "Distance left nostril"],
    "Odor_sorting_task_PEA": ["errordistances_PEA", "Dilution series PEA time required in s"],
    "Odor_sorting_task_EUG": ["errordistances_EUG", "euc Dilution series time"],
    "Lateralisation_task": ["Lat correct assignments overall"],
    "Discrimination_task": ["correct_Discrimination_task"],
    "Threshold_after_exposition": ["PEA threshold after PEA clip"]}

raw_values_of_analyzed_variables = [item for sublist in [
    a for a in groups_of_raw_values_of_analyzed_variables.values()] for item in sublist]
raw_values_of_analyzed_variables1 = set(raw_values_of_analyzed_variables).intersection(
    set(dfRiechenVerduennungAbstand.columns))

difference_vars = list(set(raw_values_of_analyzed_variables) -
                       set(dfRiechenVerduennungAbstand.columns))
raw_values_of_analyzed_variables = list(
    set(raw_values_of_analyzed_variables) - set(difference_vars))

variables_Sorting_PEA = groups_of_nongenerated_variables["Odor_sorting_task_PEA"][0:5]
dfvariables_Sorting_PEA = copy.copy(
    dfRiechenVerduennungAbstand[variables_Sorting_PEA])
variables_Sorting_EUG = groups_of_nongenerated_variables["Odor_sorting_task_EUG"][0:5]
dfvariables_Sorting_EUG = copy.copy(
    dfRiechenVerduennungAbstand_imputed[variables_Sorting_EUG])

for i, variable in enumerate(dfvariables_Sorting_PEA):
    i1 = i + 1
    dfvariables_Sorting_PEA[variable] = dfvariables_Sorting_PEA[variable].map(f)

for i, variable in enumerate(dfvariables_Sorting_EUG):
    i1 = i + 1
    dfvariables_Sorting_EUG[variable] = dfvariables_Sorting_EUG[variable].map(f)

if alternativeOderCounting:
    errordistances_PEA_raw = dfvariables_Sorting_PEA.sum(axis=1)
    errordistances_EUG_raw = dfvariables_Sorting_EUG.sum(axis=1)

else:
    errordistances_PEA_raw = abs(dfRiechenVerduennungAbstand_imputed[variables_Sorting_PEA] - [1, 2, 3, 4, 5]).sum(axis=1)
    errordistances_EUG_raw = abs(dfRiechenVerduennungAbstand_imputed[variables_Sorting_EUG] - [1, 2, 3, 4, 5]).sum(axis=1)
 
    
variables_Discrimination_task = groups_of_nongenerated_variables["Discrimination_task"]
dfvariables_Discrimination_task_raw = copy.copy(
    dfRiechenVerduennungAbstand[variables_Discrimination_task])
correct_Discrimination_task_raw = dfvariables_Discrimination_task_raw.sum(
    axis=1)

dfRiechenVerduennungAbstand_raw_values_of_analyzed_variables =  dfRiechenVerduennungAbstand[raw_values_of_analyzed_variables1].copy()

dfRiechenVerduennungAbstand_raw_values_of_analyzed_variables[
    "errordistances_PEA"] = errordistances_PEA_raw
dfRiechenVerduennungAbstand_raw_values_of_analyzed_variables[
    "errordistances_EUG"] = errordistances_EUG_raw
dfRiechenVerduennungAbstand_raw_values_of_analyzed_variables[
    "correct_Discrimination_task"] = correct_Discrimination_task_raw

X_rawplot = dfRiechenVerduennungAbstand_raw_values_of_analyzed_variables.copy()
X_rawplot.columns

variables_rename_key = {"Thr": "olfthresh", "log Thr": "log olfthresh", "Dis": "olfdis", "Id": "olfident", "ImportofO_Evaluation": "Importance of evaluation",
                        "ImportofO_Application": "Importance of application", "ImportofO_Consequence": "Importance of consequence",
                        "errordistances_PEA": "Score PEA", "errordistances_EUG": "Score EUG", 
                        "errordistances_PEA_timecorrected": "Score PEA time corrected", "errordistances_EUG_timecorrected": "Score EUG time corrected",
                        "correct_PEA": "Correct PEA", "correct_EUG": "Correct EUG",
                        "Dilution series PEA time required in s": "PEA order time", "euc Dilution series time": "EUG order time",
                        "Lat correct assignments overall": "Correct lateralisations", "correct_Discrimination_task": "Correct enantiomer discriminations",
                        "sex_0f": "Sex", "An0_Hyp1_Norm2": "Olf. diagnosis"}

X_rawplot.rename (columns = variables_rename_key ,inplace = True)
X_rawplot.columns
    
with sns.axes_style("darkgrid"):
    figData = plt.figure(figsize=(25, 8))
    gs = gridspec.GridSpec(1, 18, figure=figData, wspace=1, hspace=.1)
    # ax1 = figData.add_subplot(gs[0])
    # ax2 = figData.add_subplot(gs[1])
    ax3 = figData.add_subplot(gs[2:5])
    ax4 = figData.add_subplot(gs[5:8])
    ax5 = figData.add_subplot(gs[8:10])
    ax6 = figData.add_subplot(gs[10:12])
    ax7 = figData.add_subplot(gs[12:14])
    ax8 = figData.add_subplot(gs[14])
    ax9 = figData.add_subplot(gs[15])
    ax10 = figData.add_subplot(gs[16])

    # sns.violinplot(ax=ax1, data=X_rawplot[["Age"]], saturation=1, color = "chartreuse", linewidth=0.1)
    # ax1.set_ylabel("Years")
    # ax1.set_title("Demographics")
    # ax1.tick_params(axis="x", rotation=90)
    # sns.swarmplot(ax=ax1, data=X_rawplot[["Age"]], size = 2, color = "dodgerblue")
    # sns.violinplot(ax=ax2, data=X_rawplot[["BMI"]], saturation=1, color = "chartreuse", linewidth=0.1)
    # ax2.set_ylabel("kg/m^2")
    # #ax2.set_title("Demographics")
    # ax2.tick_params(axis="x", rotation=90)
    # sns.swarmplot(ax=ax2, data=X_rawplot[["BMI"]], size = 2, color = "dodgerblue")

    sns.violinplot(ax=ax3, data=X_rawplot[[
                   "olfthresh", "olfdis", "olfident"]], saturation=1, color="chartreuse", linewidth=0.1)
    ax3.set_ylabel("Score")
    ax3.set_title("Sniffin’ Sticks test")
    ax3.tick_params(axis="x", rotation=90)
    sns.swarmplot(
        ax=ax3, data=X_rawplot[["olfthresh", "olfdis", "olfident"]], size=3, color="dodgerblue")

    sns.violinplot(ax=ax4, data=X_rawplot[["Importance of evaluation", "Importance of application",
                   "Importance of consequence"]], saturation=1, color="chartreuse", linewidth=0.1)
    ax4.set_ylabel("Score")
    ax4.tick_params(axis="x", rotation=90)
    ax4.set_title("Importance test")
    sns.swarmplot(ax=ax4, data=X_rawplot[[
                  "Importance of evaluation", "Importance of application", "Importance of consequence"]], size=3, color="dodgerblue")

    sns.violinplot(ax=ax5, data=X_rawplot[[
                   "Score PEA", "Score EUG"]], saturation=1, color="chartreuse", linewidth=0.1)
    ax5.tick_params(axis="x", rotation=90)
    ax5.set_ylabel("Score")
    ax5.set_title("Odor order test")
    sns.swarmplot(ax=ax5, data=X_rawplot[[
                  "Score PEA", "Score EUG"]], size=3, color="dodgerblue")

    sns.violinplot(ax=ax6, data=X_rawplot[["PEA order time",
                   "EUG order time"]], saturation=1, color="chartreuse", linewidth=0.1)
    ax6.set_title("Odor order test")
    ax6.set_ylabel("s")
    ax6.tick_params(axis="x", rotation=90)
    sns.swarmplot(ax=ax6, data=X_rawplot[[
                  "PEA order time", "EUG order time"]], size=3, color="dodgerblue")

    sns.violinplot(ax=ax7, data=X_rawplot[[
                   "Distance right nostril", "Distance left nostril"]], saturation=1, color="chartreuse", linewidth=0.1)
    ax7.set_ylabel("cm")
    ax7.tick_params(axis="x", rotation=90)
    ax7.set_title("Peanut test")
    sns.swarmplot(ax=ax7, data=X_rawplot[[
                  "Distance right nostril", "Distance left nostril"]], size=3, color="dodgerblue")

    sns.violinplot(ax=ax8, data=X_rawplot[[
                   "Correct lateralisations"]], saturation=1, color="chartreuse", linewidth=0.1)
    ax8.set_ylabel("Score")
    ax8.tick_params(axis="x", rotation=90)
    ax8.set_title("Later. test")
    sns.swarmplot(ax=ax8, data=X_rawplot[[
                  "Correct lateralisations"]], size=3, color="dodgerblue")
    
    sns.violinplot(ax=ax9, data=X_rawplot[[
                   "Correct enantiomer discriminations"]], saturation=1, color="chartreuse", linewidth=0.1)
    ax9.set_ylabel("Score")
    ax9.tick_params(axis="x", rotation=90)
    ax9.set_title("Enant. test")
    sns.swarmplot(ax=ax9, data=X_rawplot[[
                  "Correct enantiomer discriminations"]], size=3, color="dodgerblue")

    sns.violinplot(ax=ax10, data=X_rawplot[[
                   "PEA threshold after PEA clip"]], saturation=1, color="chartreuse", linewidth=0.1)
    ax10.set_ylim(ax3.get_ylim())
    ax10.set_ylabel("Score")
    ax10.set_title("Adapt. test")
    ax10.tick_params(axis="x", rotation=90)
    sns.swarmplot(ax=ax10, data=X_rawplot[[
                  "PEA threshold after PEA clip"]], size=3, color="dodgerblue")

# %% Create final data for further anaylsis
FinalDataSetPreprocessed = dfRiechenVerduennungAbstand_imputed_analyzed.copy()
FinalDataSetPreprocessed.rename (columns = variables_rename_key ,inplace = True)
FinalDataSetPreprocessed.columns


# %%
