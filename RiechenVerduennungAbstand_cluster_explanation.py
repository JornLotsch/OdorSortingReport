#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 16:14:53 2022

@author: joern
"""
# %% imports

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string
from scipy import stats
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency
import matplotlib.gridspec as gridspec
from ABCanalysis import ABC_analysis

from RiechenVerduennungAbstand_correlationsPCA import DataForCorrelationPCA
from RiechenVerduennungAbstand_clustering import cluster_labels
from RiechenVerduennungAbstand_readandexploredata import dfRiechenVerduennungAbstand

# %% Functions
def cohend(d1, d2):
	# calculate the size of samples
	n1, n2 = len(d1), len(d2)
	# calculate the variance of the samples
	s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
	# calculate the pooled standard deviation
	s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	# calculate the means of the samples
	u1, u2 = np.mean(d1), np.mean(d2)
	# calculate the effect size
	return (u1 - u2) / s

def annotate_axes(ax, text, fontsize=18):
    ax.text(-.021, 1.0, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize, color="black")
    
def merge_two_dicts(x, y):
    z = x.copy()   # start with keys and values of x
    z.update(y)    # modifies z with keys and values of y
    return z
#%% Cluster explanation

df3 = DataForCorrelationPCA.copy()
df3 = pd.DataFrame(StandardScaler().fit_transform(df3), columns=df3 .columns)
n_var = len(df3.columns)
df3["cluster"] = cluster_labels
df3["cluster"].value_counts()
pvalues = [] 
for i in range(df3.shape[1]-1):
    pvalues.append(stats.mannwhitneyu(*[group[df3.columns[i]].values for name, group in df3.groupby(df3.iloc[:, -1])]).pvalue)
    print(str(df3.columns[i]) + ": " + str(pvalues[-1]))
    print(str(df3.columns[i]) + ": " + str(pvalues[-1]*n_var))

cohend_values_clusters = [] 
for i in range(df3.shape[1]-1):
    cohend_values_clusters.append(-cohend(*[group[df3.columns[i]].values for name, group in df3.groupby(df3.iloc[:, -1])]))
    print(str(df3.columns[i]) + ": " + str(cohend_values_clusters[-1]))

df3.groupby("cluster").mean().T
df3.groupby("cluster").std().T

df_cohend_clusters = pd.DataFrame(DataForCorrelationPCA.columns, columns = ["variable"])
df_cohend_clusters["Cohens' d"] = cohend_values_clusters

df_cohend_ABC = df_cohend_clusters.copy()
df_cohend_ABC["Cohens' d"] = df_cohend_ABC["Cohens' d"].abs()
df_cohend_ABC.set_index("variable",inplace=True) 
df_cohend_ABC.sort_values(by = "Cohens' d", inplace=True, ascending = False)
ABC_analysis(data = df_cohend_ABC["Cohens' d"])

df3_long = pd.melt(df3, "cluster", var_name="variable", value_name="value")

with sns.axes_style("darkgrid"):
    fig = plt.figure(figsize=(25, 8))
    gs0 = gridspec.GridSpec(15, 4, figure=fig, wspace=.1, hspace=1)

    ax1 = fig.add_subplot(gs0[4:,:3])  
    ax2 = fig.add_subplot(gs0[:4,:3])  
    ax3 = fig.add_subplot(gs0[:4,3])  
    ax4 = fig.add_subplot(gs0[4:,3])  
    axes = [ax2, ax1, ax3, ax4]
    for i, ax in enumerate(axes):
        annotate_axes(ax,  str(string.ascii_lowercase[i])+ ")")
                
    sns.violinplot(ax=ax1, data=df3_long , x = "variable", y = "value", hue = "cluster", saturation=1, linewidth= 0.1,
                   palette=["dodgerblue", "chartreuse"])    
    g = sns.swarmplot(ax=ax1, data=df3_long , x = "variable", y = "value", hue = "cluster", dodge = "hue", size = 3, palette=["blue", "green"])
    textColors =[("red" if i <  0.05/n_var  else ("blue" if i < 0.05 else "black")) for i in pvalues]
    for i in range(n_var):
        ax1.text(x=i, y = 0.8 * ax1.get_ylim()[1], s = "{:.2e}".format(pvalues[i]), rotation = 90, color = textColors[i], va="center", ha="center")
    ax1.set_xlabel(None)
    ax1.set_ylabel("z-Value")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
    g.legend(loc='lower left')

    ABC_A_varimportance_Cohen = ABC_analysis(ax = ax4, data = df_cohend_ABC["Cohens' d"], PlotIt = True)["ABlimit"]
    barcols = ["dodgerblue" if abs(i) < ABC_A_varimportance_Cohen else "blue" for i in df_cohend_clusters["Cohens' d"]]
    sns.barplot(ax=ax2, data = df_cohend_clusters, x = "variable", y = "Cohens' d", palette = barcols, alpha = .5)
    ax2.set_xlabel(None)
    ax2.set_xticklabels([])
    ax2.axhline(0, linewidth=1, color="grey", linestyle = "dashed")
    ax2.axhline(.8, linewidth=1, color="salmon", linestyle = "dotted")
    ax2.axhline(.5, linewidth=1, color="salmon", linestyle = "dotted")
    ax2.axhline(.2, linewidth=1, color="salmon", linestyle = "dotted")
    ax2.axhline(-.2, linewidth=1, color="salmon", linestyle = "dotted")
    ax2.axhline(-.5, linewidth=1, color="salmon", linestyle = "dotted")
    ax2.axhline(-.8, linewidth=1, color="salmon", linestyle = "dotted")
    ax2.text(0, .9, "Large", color="red")
    ax2.text(0, .6, "Medium", color="red")
    ax2.text(0, .3, "Small", color="red")
    ax2.text(0, -.4, "Small", color="red")
    ax2.text(0, -.7, "Medium", color="red")
    ax2.text(0, -1, "Large", color="red")

    dfClusterOlfdiag = df3[["cluster"]].copy()
    dfClusterOlfdiag["Olfdiag"] = dfRiechenVerduennungAbstand["An0_Hyp1_Norm2"]

    props1 = {}
    for x in ['1']:
        for y, col in {'0': 'royalblue', '1': 'limegreen'}.items():
            props1[(y, x)] ={'color': col}
    props2 = {}
    for x in ['2']:
        for y, col in {'0': 'dodgerblue', '1': 'chartreuse'}.items():
            props2[(y, x)] ={'color': col}
    
    props = merge_two_dicts(props1, props2)

    mosaic(dfClusterOlfdiag, ["cluster" ,"Olfdiag"], gap=0.05, properties=props, ax = ax3)
    ax3.set_xlabel("Cluster")
    ax3.set_ylabel("Diagnosis")

    contigency_1= pd.crosstab(dfClusterOlfdiag['Olfdiag'], dfClusterOlfdiag["cluster"])
    contigency_1
    c, p, dof, expected = chi2_contingency(contigency_1)
    ax3.text(.3, .6, "Chi square: p = " + "{:.4f}".format(p), color="darkred")
    ax3.xaxis.set_ticks_position("top")

#%% Cluster explanation comparison with olfactory diagnoses or sex

dfClusterOlfdiagSex = dfClusterOlfdiag.copy()
dfClusterOlfdiagSex['sex_0f'] = dfRiechenVerduennungAbstand['sex_0f']

contigency_3= pd.crosstab(dfClusterOlfdiagSex['sex_0f'], dfClusterOlfdiagSex["cluster"])
contigency_3
c, p, dof, expected = chi2_contingency(contigency_3)

contigency_5= pd.crosstab(dfClusterOlfdiagSex['Olfdiag'], dfClusterOlfdiagSex["sex_0f"])
contigency_5
c, p, dof, expected = chi2_contingency(contigency_5)

df_normosmics = df3.copy()[(dfClusterOlfdiag["Olfdiag"] == 2) ]

df4 = DataForCorrelationPCA.copy()
df4 = pd.DataFrame(StandardScaler().fit_transform(df4), columns=df4 .columns)
n_var = len(df4.columns)
df4["Olfdiag"] = dfClusterOlfdiag["Olfdiag"]

cohend_values_olfdiag = [] 
for i in range(df4.shape[1]-1):
    cohend_values_olfdiag.append(-cohend(*[group[df4.columns[i]].values for name, group in df4.groupby(df4.iloc[:, -1])]))
    print(str(df4.columns[i]) + ": " + str(cohend_values_olfdiag[-1]))

df_cohend_clusters_olfdiag = df_cohend_clusters.copy()
df_cohend_clusters_olfdiag.rename(columns = {"Cohens' d": "d Cluster"}, inplace = True)
df_cohend_clusters_olfdiag["d OlfDiag"] = cohend_values_olfdiag
df_cohend_clusters_olfdiag["d_cluster_not_olfdiag"] = abs(df_cohend_clusters_olfdiag["d Cluster"]) - abs(df_cohend_clusters_olfdiag["d OlfDiag"])

df_cohend_ABC_d_cluster_not_olfdiag = df_cohend_clusters_olfdiag[["variable", "d_cluster_not_olfdiag"]]
df_cohend_ABC_d_cluster_not_olfdiag .set_index("variable",inplace=True) 
df_cohend_ABC_d_cluster_not_olfdiag .sort_values(by = "d_cluster_not_olfdiag", inplace=True, ascending = False)



with sns.axes_style("darkgrid"):
    fig = plt.figure(figsize=(12, 14))
    gs0 = gridspec.GridSpec(2, 2, figure=fig, wspace=.1, hspace=0.4)

    ax2 = fig.add_subplot(gs0[0,0])  
    ax1 = fig.add_subplot(gs0[0,1])  
    axes = [ax2, ax1]
    for i, ax in enumerate(axes):   
        annotate_axes(ax,  str(string.ascii_lowercase[i])+ ")")

    
    ABC_A_varimportance_d_cluster_not_olfdiag = ABC_analysis(ax = ax1, data = df_cohend_ABC_d_cluster_not_olfdiag ["d_cluster_not_olfdiag"], PlotIt = True)
    ax1.set_title("ABC plot")

    barcols = ["dodgerblue" if (i) < ABC_A_varimportance_d_cluster_not_olfdiag["ABlimit"]  else "blue" for i in df_cohend_ABC_d_cluster_not_olfdiag["d_cluster_not_olfdiag"]]
    sns.barplot(ax=ax2, data = df_cohend_ABC_d_cluster_not_olfdiag, y = df_cohend_ABC_d_cluster_not_olfdiag.index.tolist(), x = "d_cluster_not_olfdiag", palette = barcols, alpha = 1)
    ax2.set_title("Difference of effect sizes")
    ax2.set_xlabel("|Cohen's d for clusters| - |Cohen's d for olfactory diagnosis|")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
