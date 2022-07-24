#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 16:14:53 2022

@author: joern
"""
# %% imports

import pandas as pd
import numpy as np
from sklearn.base import clone
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples, normalized_mutual_info_score, adjusted_mutual_info_score
import matplotlib.gridspec as gridspec
from silhouette_plots import silhouette_plots
from scipy.spatial import ConvexHull
import string

from RiechenVerduennungAbstand_correlationsPCA import PCA_data

# %%Functions


def annotate_axes(ax, text, fontsize=18):
    ax.text(-.021, .95, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize, color="black")


def cluster_stability(X, cluster_labels, est, n_iter=20, random_state=None):
    rng = np.random.RandomState(6)
    ari, nmi, ami, sil = [], [], [], []

    for i in range(n_iter):
        sample_indices = rng.randint(0, X.shape[0], X.shape[0])
        est = clone(est)
        if hasattr(est, "random_state"):
            # randomize estimator if possible
            est.random_state = rng.randint(1e5)
        X_bootstrap = X[sample_indices]
        y_orig_bootstrap = cluster_labels[sample_indices]
        est.fit(X_bootstrap)
        y_pred_bootstrap = est.fit_predict(X_bootstrap)

        ari.append(adjusted_rand_score(y_orig_bootstrap, y_pred_bootstrap))
        nmi.append(normalized_mutual_info_score(
            y_orig_bootstrap, y_pred_bootstrap))
        ami.append(adjusted_mutual_info_score(
            y_orig_bootstrap, y_pred_bootstrap))
        sil.append(silhouette_score(X_bootstrap, y_pred_bootstrap))
    return ari, nmi, ami, sil


# %% Clustering
PCA_data.columns
pca2 = PCA(7)
PCA_data_projected = pca2.fit_transform(PCA_data)
dfPCA_data_projected = pd.DataFrame(PCA_data_projected)

file = "/home/joern/PCA_data.csv"
PCA_data.to_csv(file)
file = "/home/joern/PCA_data_projected.csv"
dfPCA_data_projected.to_csv(file)

df = dfPCA_data_projected.copy()
df .columns = ["PC" + str(i) for i in range(1, 8)]

kmeans = KMeans(random_state=0)
silhouette_plots(data=dfPCA_data_projected, est=kmeans, random_state=0)
ward = AgglomerativeClustering()
silhouette_plots(data=dfPCA_data_projected, est=ward, random_state=0)

kmeans = KMeans(n_clusters=2, random_state=0)
cluster_labels = kmeans.fit_predict(df)
df["cluster"] = cluster_labels
n_cluster = len(df["cluster"].unique())

df["cluster"].value_counts()

centroids = kmeans.cluster_centers_
cen_x = [i[0] for i in centroids]
cen_y = [i[1] for i in centroids]

df['cen_x'] = [cen_x[i] for i in df["cluster"].values]
df['cen_y'] = [cen_y[i] for i in df["cluster"].values]
# df['cen_x'] = df.cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2]})
# df['cen_y'] = df.cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2]})

colors = ['dodgerblue', '#4b9c00', '#2095DF']
df['c'] = [colors[i] for i in df["cluster"].values]
#df['c'] = df.cluster.map({0:colors[0], 1:colors[1], 2:colors[2]})

# %% Plot clusters
# https://towardsdatascience.com/visualizing-clusters-with-pythons-matplolib-35ae03d87489

with sns.axes_style("darkgrid"):
    fig = plt.figure(figsize=(15, 10))
    gs0 = gridspec.GridSpec(1, 3, figure=fig, wspace=.2, hspace=.2)

    ax1 = fig.add_subplot(gs0[0, :2])
    ax2 = fig.add_subplot(gs0[0, 2])
    axes = [ax1, ax2]
    for i, ax in enumerate(axes):
        annotate_axes(ax,  str(string.ascii_lowercase[i]) + ")")

    sns.scatterplot(ax=ax1, x=df.iloc[:, 0], y=df.iloc[:, 1], c=df["c"])
    sns.scatterplot(ax=ax1, x=cen_x, y=cen_y, marker='^',
                    c=colors[:n_cluster], s=70)
    ax1.set_title("Factorial plot")
    for idx, val in df.iterrows():
        x = [val.iloc[0], val.cen_x, ]
        y = [val.iloc[1], val.cen_y]
        sns.lineplot(ax=ax1, x=x, y=y, color=val.c, alpha=0.2)
    for i in df.cluster.unique():
        points = df[df.cluster == i][['PC1', 'PC2']].values
        # get convex hull
        hull = ConvexHull(points)
        x_hull = np.append(points[hull.vertices, 0],
                           points[hull.vertices, 0][0])
        y_hull = np.append(points[hull.vertices, 1],
                           points[hull.vertices, 1][0])
        ax1.fill_between(x_hull, y_hull, alpha=0.3, fc=colors[i])

    sample_silhouette_values = silhouette_samples(
        dfPCA_data_projected, df["cluster"])
    y_lower = 10
    for i in range(n_cluster):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[df["cluster"] == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

       # color = cm.nipy_spectral(float(i) / 3)
        ax2.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=colors[i],
            edgecolor=colors[i],
            alpha=0.7,
        )

        ax2.set_title("Silhouette  plot")
        # Label the silhouette plots with their cluster numbers at the middle
        ax2.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        ax2.set_xlabel("Silhouette width")
        ax2.set_ylabel("Subject count")

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    mean_sil = np.mean(sample_silhouette_values)
    ax2.axvline(mean_sil, color="salmon", linestyle="dotted")
    ax2.text(1.05*mean_sil, 0.9 * ax2.get_ylim()[1], "Average silhouette width: " + "{:.3f}".format(mean_sil),
             rotation=90, va="center_baseline", color="black")
# %% Cluster stability etc

df2 = df.iloc[:, :7].to_numpy()

#cluster_stability(X= df2, est = kmeans, n_iter=20, random_state=None)


ari, nmi, ami, sil = cluster_stability(
    X=df2, cluster_labels=cluster_labels, est=kmeans, n_iter=20)
# print(np.percentile(ari, (2.5,50,97.5)))
# print(np.percentile(nmi, (2.5,50,97.5)))
# print(np.percentile(ami, (2.5,50,97.5)))
# print(np.percentile(sil, (2.5,50,97.5)))

print(np.mean(sil))
print(np.mean(ari))
