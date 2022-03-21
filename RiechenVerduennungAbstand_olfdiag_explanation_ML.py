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
import matplotlib.gridspec as gridspec
import string
from numpy import mean
from numpy import std

from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel, RFE, SequentialFeatureSelector, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from ABCanalysis import ABC_analysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline

from RiechenVerduennungAbstand_correlationsPCA import DataForCorrelationPCA, PCA_features
from RiechenVerduennungAbstand_cluster_explanation import dfClusterOlfdiagSex
from RiechenVerduennungAbstand_cluster_explanation_ML import FS_sumscore_cluster

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


# %% Get data generated in earlier anaylses
df5 = DataForCorrelationPCA.copy()
df5.columns
#Min_Max = MinMaxScaler()
Standardize = StandardScaler()
df5 = pd.DataFrame(Standardize .fit_transform(df5), columns=df5.columns)

df5['sex_0f'] = dfClusterOlfdiagSex['sex_0f']
df5['Olfdiag'] = dfClusterOlfdiagSex["Olfdiag"]
df5.columns

# %% OlfDiag explnation ML

y = df5["Olfdiag"]
pd.DataFrame(y).value_counts()
df5.drop(["Olfdiag"], axis=1, inplace=True)
df5.columns


X_train, X_test, y_train, y_test = train_test_split(
    df5, y, test_size=0.2, random_state=42)
pd.DataFrame(y_test).value_counts()
# https://towardsdatascience.com/how-to-find-decision-tree-depth-via-cross-validation-2bf143f0f3d6

#%% Classifier tuning 

# LinearSVC
lsvc = LinearSVC(max_iter=10000)
param_grid = {"C": np.arange(0.01,100,10), 
              "penalty": ["l1", "l2"], 
              "dual": [True, False], 
              "loss": ["hinge", "squared_hinge"],
              "tol": [0.001,0.0001,0.00001]}
grid_search = GridSearchCV(lsvc, param_grid=param_grid, scoring="balanced_accuracy", verbose=0, n_jobs = -1)
grid_search.fit(X_train, y_train)
C_lsvm, dual_svm, loss_svm, penalty_SVM, tol_svm = grid_search.best_params_.values()

# Random forests
forest = RandomForestClassifier(random_state=0)
param_grid = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
grid_search = GridSearchCV(forest, param_grid=param_grid, scoring="balanced_accuracy", verbose=0, n_jobs = -1)
grid_search.fit(X_train, y_train)
bootstrap_rf,  max_depth_rf, max_features_rf, min_samples_leaf_rf, min_samples_split_rf, n_estimators_rf = grid_search.best_params_.values()

# Logistic regregssion
LogReg = LogisticRegression(max_iter=10000, random_state=0)
param_grid ={"C":np.logspace(-3,3,7),
             "penalty":["l1", "l2", "elasticnet"],
             "tol": [0.001,0.0001,0.00001],
             "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]}
grid_search = GridSearchCV(LogReg, param_grid=param_grid, scoring="balanced_accuracy", verbose=0, n_jobs = -1)
grid_search.fit(X_train, y_train)
C_LogReg,  penalty_LogReg, solver_LogReg, tol_LogReg= grid_search.best_params_.values()

# %% Create results table all methods and variables
featureSelection_methods = ["PCA", "features_CohenD", 
                            "features_lSVC_sKb", "features_RF_sKb", "features_LogReg_sKb", 
                            "features_lSVC_sfm", "features_RF_sfm", "features_LogReg_sfm",
                            "features_lSVC_rfe", "features_RF_rfe", "features_LogReg_rfe",
                            "features_lSVC_sfs_forward", "features_RF_sfs_forward", "features_LogReg_sfs_forward",
                            "features_lSVC_sfs_backward", "features_RF_sfs_backward", "features_LogReg_sfs_backward"]
                            
feature_table = pd.DataFrame(np.zeros((len(X_train.columns),len(featureSelection_methods))))
feature_table.columns = featureSelection_methods
feature_table.set_index(X_train.columns, inplace=True)

# Add PCA results from previous anaylsis

feature_table.loc[PCA_features["Aind"].index.tolist(), "PCA"] = 1
                       
# %% CV
n_splits = 5
n_repeats = 20

#%% Cohens's d CV

features_CohenD = []

rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)
for train_index, test_index in rskf.split(X_train, y_train):
    X_train_FS, X_test_FS = X_train.iloc[train_index,
                                         :], X_train.iloc[test_index, :]
    y_train_FS, y_test_FS = y_train.iloc[train_index], y_train.iloc[test_index]

    X_train_FS_CohenD = X_train_FS.copy()
    X_train_FS_CohenD["y"] = y_train_FS
    
    chd = [] 
    for i1 in range(X_train_FS_CohenD.shape[1]-1):
        chd .append(abs(cohend(*[group[X_train_FS_CohenD.columns[i1]].values for name, group in X_train_FS_CohenD.groupby(X_train_FS_CohenD.iloc[:, -1])])))
    
    
    df_chd= pd.DataFrame(X_train_FS.columns, columns = ["variable"])
    df_chd.set_index("variable",inplace=True) 

    df_chd["Cohens' d"] = chd
    features_CohenD.append(ABC_analysis(data = df_chd["Cohens' d"])["Aind"].index)
    
features_CohenD_all = []
for i in range(len(features_CohenD)):
    for j in range(len(features_CohenD[i])):
        features_CohenD_all.append(features_CohenD[i][j])
features_CohenD_all = pd.DataFrame({"Counts":  pd.DataFrame(features_CohenD_all).value_counts()})
features_CohenD_all.reset_index()
ABCres = ABC_analysis(features_CohenD_all.iloc[:,0])
ABCres["Aind"].reset_index(inplace=True)
feature_table.loc[ABCres["Aind"].iloc[:,0].tolist(), "features_CohenD"] = 1

fig, ax = plt.subplots(figsize=(18, 16))
ax = sns.barplot(x=features_CohenD_all.index.to_numpy(), y=features_CohenD_all["Counts"])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

# %% Feature selection univariate selectKbest

features_lSVC_sKb = []
features_RF_sKb = []
features_LogReg_sKb = []
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)
for train_index, test_index in rskf.split(X_train, y_train):
    X_train_FS, X_test_FS = X_train.iloc[train_index,
                                         :], X_train.iloc[test_index, :]
    y_train_FS, y_test_FS = y_train.iloc[train_index], y_train.iloc[test_index]
    
    
    anova_filter = SelectKBest(f_classif, k=3)
    lsvc = LinearSVC(C = C_lsvm, penalty = penalty_SVM, dual = dual_svm, loss = loss_svm, tol = tol_svm, max_iter=10000)
    pipeline = Pipeline([("anova_filter", anova_filter ), ("lsvc", lsvc)])
    param_grid = dict(
        anova_filter__k=list(range(1,X_train_FS.shape[1])),
        )
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring="balanced_accuracy", verbose=0,n_jobs = -1)
    grid_search.fit(X_train_FS, y_train_FS)
    k = max(grid_search.best_params_.values())

    X_new = SelectKBest(f_classif, k = k).fit(X_train_FS, y_train_FS)
    features_lSVC_sKb.append(X_new.get_feature_names_out().tolist())

    forest = RandomForestClassifier(random_state=0, bootstrap=bootstrap_rf,  max_depth=max_depth_rf, max_features=max_features_rf, 
                                    min_samples_leaf=min_samples_leaf_rf, min_samples_split=min_samples_split_rf, n_estimators=n_estimators_rf)
    pipeline = Pipeline([("anova_filter", anova_filter ), ("forest", forest)])
    param_grid = dict(
        anova_filter__k=list(range(1,X_train_FS.shape[1])),
    )
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring="balanced_accuracy", verbose=0,n_jobs = -1)
    grid_search.fit(X_train_FS, y_train_FS)
    k = max(grid_search.best_params_.values())

    X_new = SelectKBest(f_classif, k = k).fit(X_train_FS, y_train_FS)
    features_RF_sKb.append(X_new.get_feature_names_out().tolist())

    LogReg = LogisticRegression(C=C_LogReg,  penalty=penalty_LogReg, solver= solver_LogReg, tol=tol_LogReg, max_iter=10000,random_state=0)
    pipeline = Pipeline([("anova_filter", anova_filter ), ("LogReg", LogReg)])
    param_grid = dict(
        anova_filter__k=list(range(1,X_train_FS.shape[1])),
    )
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring="balanced_accuracy", verbose=0,n_jobs = -1)
    grid_search.fit(X_train_FS, y_train_FS)
    k = max(grid_search.best_params_.values())

    X_new = SelectKBest(f_classif, k = k).fit(X_train_FS, y_train_FS)
    features_LogReg_sKb.append(X_new.get_feature_names_out().tolist())
    
features_lSVC_sKb_all = []
for i in range(len(features_lSVC_sKb)):
    for j in range(len(features_lSVC_sKb[i])):
        features_lSVC_sKb_all.append(features_lSVC_sKb[i][j])
features_lSVC_sKb_all = pd.DataFrame({"Counts":  pd.DataFrame(features_lSVC_sKb_all).value_counts()})
features_lSVC_sKb_all.reset_index()
ABCres = ABC_analysis(features_lSVC_sKb_all.iloc[:,0])
ABCres["Aind"].reset_index(inplace=True)
feature_table.loc[ABCres["Aind"].iloc[:,0].tolist(), "features_lSVC_sKb"] = 1

features_RF_sKb_all = []
for i in range(len(features_RF_sKb)):
    for j in range(len(features_RF_sKb[i])):
        features_RF_sKb_all.append(features_RF_sKb[i][j])
features_RF_sKb_all = pd.DataFrame({"Counts":  pd.DataFrame(features_RF_sKb_all).value_counts()})
features_RF_sKb_all.reset_index()
ABCres = ABC_analysis(features_RF_sKb_all.iloc[:,0])
ABCres["Aind"].reset_index(inplace=True)
feature_table.loc[ABCres["Aind"].iloc[:,0].tolist(), "features_RF_sKb"] = 1

features_LogReg_sKb_all = []
for i in range(len(features_LogReg_sKb)):
    for j in range(len(features_LogReg_sKb[i])):
        features_LogReg_sKb_all.append(features_LogReg_sKb[i][j])
features_LogReg_sKb_all = pd.DataFrame({"Counts":  pd.DataFrame(features_LogReg_sKb_all).value_counts()})
features_LogReg_sKb_all.reset_index()
ABCres = ABC_analysis(features_LogReg_sKb_all.iloc[:,0])
ABCres["Aind"].reset_index(inplace=True)
feature_table.loc[ABCres["Aind"].iloc[:,0].tolist(), "features_LogReg_sKb"] = 1

features_sKb = pd.concat({"SVM": features_lSVC_sKb_all,
                         "RF": features_RF_sKb_all, "LogReg": features_LogReg_sKb_all}, axis=1)
variablenames = []
for i in range(len(features_sKb .index)):
    variablenames.append(features_sKb .index[i][0])
features_sKb["variable"] = (variablenames)
features_sKb.set_index("variable", inplace=True)

fig, ax = plt.subplots(figsize=(18, 16))
ax = sns.barplot(x=features_sKb.index.tolist(), y=features_sKb.iloc[:, 0])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

# %% Feature selection Select from model

features_lSVC_sfm = []
features_RF_sfm = []
features_LogReg_sfm = []
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)
for train_index, test_index in rskf.split(X_train, y_train):
    X_train_FS, X_test_FS = X_train.iloc[train_index,
                                         :], X_train.iloc[test_index, :]
    y_train_FS, y_test_FS = y_train.iloc[train_index], y_train.iloc[test_index]

    lsvc = LinearSVC(C = C_lsvm, penalty = penalty_SVM, dual = dual_svm, loss = loss_svm, tol = tol_svm, max_iter=10000).fit(
        X_train_FS, y_train_FS)  # was 0.01 for nonscaled data
    model_lsvc = SelectFromModel(lsvc, prefit=True)
    feature_idx = model_lsvc.get_support()
    feature_name = X_train.columns[feature_idx]
    features_lSVC_sfm.append(feature_name)

    forest = RandomForestClassifier(random_state=0, bootstrap=bootstrap_rf,  max_depth=max_depth_rf, max_features=max_features_rf, 
                                    min_samples_leaf=min_samples_leaf_rf, min_samples_split=min_samples_split_rf, n_estimators=n_estimators_rf).fit(X_train_FS, y_train_FS)
    model_forest = SelectFromModel(forest, prefit=True)
    feature_idx = model_forest.get_support()
    feature_name = X_train.columns[feature_idx]
    features_RF_sfm.append(feature_name)

    LogReg = LogisticRegression(
        C=C_LogReg,  penalty=penalty_LogReg, solver= solver_LogReg, tol=tol_LogReg, max_iter=10000,random_state=0).fit(X_train_FS, y_train_FS)
    model_reg = SelectFromModel(LogReg, prefit=True)
    feature_idx = model_reg.get_support()
    feature_name = X_train.columns[feature_idx]
    features_LogReg_sfm.append(feature_name)

features_lSVC_sfm_all = []
for i in range(len(features_lSVC_sfm)):
    for j in range(len(features_lSVC_sfm[i])):
        features_lSVC_sfm_all.append(features_lSVC_sfm[i][j])
features_lSVC_sfm_all = pd.DataFrame({"Counts":  pd.DataFrame(features_lSVC_sfm_all).value_counts()})
features_lSVC_sfm_all.reset_index()
ABCres = ABC_analysis(features_lSVC_sfm_all.iloc[:,0])
ABCres["Aind"].reset_index(inplace=True)
feature_table.loc[ABCres["Aind"].iloc[:,0].tolist(), "features_lSVC_sfm"] = 1

features_RF_sfm_all = []
for i in range(len(features_RF_sfm)):
    for j in range(len(features_RF_sfm[i])):
        features_RF_sfm_all.append(features_RF_sfm[i][j])
features_RF_sfm_all = pd.DataFrame({"Counts":  pd.DataFrame(features_RF_sfm_all).value_counts()})
features_RF_sfm_all.reset_index()
ABCres = ABC_analysis(features_RF_sfm_all.iloc[:,0])
ABCres["Aind"].reset_index(inplace=True)
feature_table.loc[ABCres["Aind"].iloc[:,0].tolist(), "features_RF_sfm"] = 1

features_LogReg_sfm_all = []
for i in range(len(features_LogReg_sfm)):
    for j in range(len(features_LogReg_sfm[i])):
        features_LogReg_sfm_all.append(features_LogReg_sfm[i][j])
features_LogReg_sfm_all = pd.DataFrame({"Counts":  pd.DataFrame(features_LogReg_sfm_all).value_counts()})
features_LogReg_sfm_all.reset_index()
ABCres = ABC_analysis(features_LogReg_sfm_all.iloc[:,0])
ABCres["Aind"].reset_index(inplace=True)
feature_table.loc[ABCres["Aind"].iloc[:,0].tolist(), "features_LogReg_sfm"] = 1

features_SFM = pd.concat({"SVM": features_lSVC_sfm_all,
                         "RF": features_RF_sfm_all, "LogReg": features_LogReg_sfm_all}, axis=1)
variablenames = []
for i in range(len(features_SFM .index)):
    variablenames.append(features_SFM .index[i][0])
features_SFM["variable"] = (variablenames)
features_SFM.set_index("variable", inplace=True)

fig, ax = plt.subplots(figsize=(18, 16))
ax = sns.barplot(x=features_SFM.index.tolist(), y=features_SFM.iloc[:, 0])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

# %% Feature selection RFE

i = 0
features_lSVC_rfe = pd.DataFrame(np.zeros(
    (len(X_train.columns), n_splits*n_repeats)), index=X_train.columns.tolist())
features_RF_rfe = pd.DataFrame(np.zeros(
    (len(X_train.columns), n_splits*n_repeats)), index=X_train.columns.tolist())
features_LogReg_rfe = pd.DataFrame(np.zeros(
    (len(X_train.columns), n_splits*n_repeats)), index=X_train.columns.tolist())
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)
for train_index, test_index in rskf.split(X_train, y_train):
    X_train_FS, X_test_FS = X_train.iloc[train_index,
                                         :], X_train.iloc[test_index, :]
    y_train_FS, y_test_FS = y_train.iloc[train_index], y_train.iloc[test_index]

    lsvc = LinearSVC(C = C_lsvm, penalty = penalty_SVM, dual = dual_svm, loss = loss_svm, tol = tol_svm, max_iter=10000)
    rfe = RFE(estimator=lsvc, n_features_to_select=1, step=1)
    rfe.fit(X_train_FS, y_train_FS)
    ranking = rfe.ranking_
    ranking = max(ranking) - ranking
    features_lSVC_rfe.iloc[:, i] = ranking

    forest = RandomForestClassifier(random_state=0, bootstrap=bootstrap_rf,  max_depth=max_depth_rf, max_features=max_features_rf, 
                                    min_samples_leaf=min_samples_leaf_rf, min_samples_split=min_samples_split_rf, n_estimators=n_estimators_rf)
    rfe = RFE(estimator=forest, n_features_to_select=1, step=1)
    rfe.fit(X_train_FS, y_train_FS)
    ranking = rfe.ranking_
    ranking = max(ranking) - ranking
    features_RF_rfe.iloc[:, i] = ranking

    LogReg = LogisticRegression(C=C_LogReg,  penalty=penalty_LogReg, solver= solver_LogReg, tol=tol_LogReg, max_iter=10000,random_state=0)
    rfe = RFE(estimator=LogReg, n_features_to_select=1, step=1)
    rfe.fit(X_train_FS, y_train_FS)
    ranking = rfe.ranking_
    ranking = max(ranking) - ranking
    features_LogReg_rfe.iloc[:, i] = ranking

    i += 1

features_lSVC_rfe_all = features_lSVC_rfe.sum(
    axis=1).sort_values(ascending=False)
ABCres = ABC_analysis(features_lSVC_rfe_all)
feature_table.loc[ABCres["Aind"].index.tolist(), "features_lSVC_rfe"] = 1

features_RF_rfe_all = features_RF_rfe.sum(axis=1).sort_values(ascending=False)
ABCres = ABC_analysis(features_RF_rfe_all)
feature_table.loc[ABCres["Aind"].index.tolist(), "features_RF_rfe"] = 1

features_LogReg_rfe_all = features_LogReg_rfe.sum(
    axis=1).sort_values(ascending=False)
ABCres = ABC_analysis(features_LogReg_rfe_all)
feature_table.loc[ABCres["Aind"].index.tolist(), "features_LogReg_rfe"] = 1

features_RFE = pd.concat({"SVM": features_lSVC_rfe_all,
                         "RF": features_RF_rfe_all, "LogReg": features_LogReg_rfe_all}, axis=1)
fig, ax = plt.subplots(figsize=(18, 16))
ax = sns.barplot(x=features_RFE.index.tolist(), y=features_RFE.iloc[:, 0])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

# %% Feature selection SFS forward

features_lSVC_sfs_forward = []
features_RF_sfs_forward = []
features_LogReg_sfs_forward = []
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)
for train_index, test_index in rskf.split(X_train, y_train):
    X_train_FS, X_test_FS = X_train.iloc[train_index,
                                         :], X_train.iloc[test_index, :]
    y_train_FS, y_test_FS = y_train.iloc[train_index], y_train.iloc[test_index]

    lsvc = LinearSVC(C = C_lsvm, penalty = penalty_SVM, dual = dual_svm, loss = loss_svm, tol = tol_svm, max_iter=10000)
    sfs = SequentialFeatureSelector(lsvc, direction="forward", n_jobs=-1)
    sfs.fit(X_train_FS, y_train_FS)
    feature_idx = sfs.get_support()
    feature_name = X_train.columns[feature_idx]
    features_lSVC_sfs_forward.append(feature_name)

    forest = RandomForestClassifier(random_state=0, bootstrap=bootstrap_rf,  max_depth=max_depth_rf, max_features=max_features_rf, 
                                    min_samples_leaf=min_samples_leaf_rf, min_samples_split=min_samples_split_rf, n_estimators=n_estimators_rf)
    sfs = SequentialFeatureSelector(forest, direction="forward", n_jobs=-1)
    sfs.fit(X_train_FS, y_train_FS)
    feature_idx = sfs.get_support()
    feature_name = X_train.columns[feature_idx]
    features_RF_sfs_forward.append(feature_name)

    LogReg = LogisticRegression(C=C_LogReg,  penalty=penalty_LogReg, solver= solver_LogReg, tol=tol_LogReg, max_iter=10000,random_state=0)
    sfs = SequentialFeatureSelector(LogReg, direction="forward", n_jobs=-1)
    sfs.fit(X_train_FS, y_train_FS)
    feature_idx = sfs.get_support()
    feature_name = X_train.columns[feature_idx]
    features_LogReg_sfs_forward.append(feature_name)


features_lSVC_sfs_forward_all = []
for i in range(len(features_lSVC_sfs_forward)):
    for j in range(len(features_lSVC_sfs_forward[i])):
        features_lSVC_sfs_forward_all.append(features_lSVC_sfs_forward[i][j])
features_lSVC_sfs_forward_all = pd.DataFrame({"Counts":  pd.DataFrame(features_lSVC_sfs_forward_all).value_counts()})
features_lSVC_sfs_forward_all.reset_index()
ABCres = ABC_analysis(features_lSVC_sfs_forward_all.iloc[:,0])
ABCres["Aind"].reset_index(inplace=True)
feature_table.loc[ABCres["Aind"].iloc[:,0].tolist(), "features_lSVC_sfs_forward"] = 1

features_RF_sfs_forward_all = []
for i in range(len(features_RF_sfs_forward)):
    for j in range(len(features_RF_sfs_forward[i])):
        features_RF_sfs_forward_all.append(features_RF_sfs_forward[i][j])
features_RF_sfs_forward_all = pd.DataFrame({"Counts":  pd.DataFrame(features_RF_sfs_forward_all).value_counts()})
features_RF_sfs_forward_all.reset_index()
ABCres = ABC_analysis(features_RF_sfs_forward_all.iloc[:,0])
ABCres["Aind"].reset_index(inplace=True)
feature_table.loc[ABCres["Aind"].iloc[:,0].tolist(), "features_RF_sfs_forward"] = 1

features_LogReg_sfs_forward_all = []
for i in range(len(features_LogReg_sfs_forward)):
    for j in range(len(features_LogReg_sfs_forward[i])):
        features_LogReg_sfs_forward_all.append(
            features_LogReg_sfs_forward[i][j])
features_LogReg_sfs_forward_all = pd.DataFrame({"Counts":  pd.DataFrame(features_LogReg_sfs_forward_all).value_counts()})
features_LogReg_sfs_forward_all.reset_index()
ABCres = ABC_analysis(features_LogReg_sfs_forward_all.iloc[:,0])
ABCres["Aind"].reset_index(inplace=True)
feature_table.loc[ABCres["Aind"].iloc[:,0].tolist(), "features_LogReg_sfs_forward"] = 1

features_sfs_forward = pd.concat({"SVM": features_lSVC_sfs_forward_all,
                                 "RF": features_RF_sfs_forward_all, "LogReg": features_LogReg_sfs_forward_all}, axis=1)
variablenames = []
for i in range(len(features_sfs_forward .index)):
    variablenames.append(features_sfs_forward .index[i][0])
features_sfs_forward["variable"] = (variablenames)
features_sfs_forward.set_index("variable", inplace=True)

fig, ax = plt.subplots(figsize=(18, 16))
ax = sns.barplot(x=features_sfs_forward.index.tolist(),
                 y=features_sfs_forward.iloc[:, 0])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)


# %% Feature selection SFS backward

features_lSVC_sfs_backward = []
features_RF_sfs_backward = []
features_LogReg_sfs_backward = []
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)
for train_index, test_index in rskf.split(X_train, y_train):
    X_train_FS, X_test_FS = X_train.iloc[train_index,
                                         :], X_train.iloc[test_index, :]
    y_train_FS, y_test_FS = y_train.iloc[train_index], y_train.iloc[test_index]

    lsvc = LinearSVC(C = C_lsvm, penalty = penalty_SVM, dual = dual_svm, loss = loss_svm, tol = tol_svm, max_iter=10000)
    sfs = SequentialFeatureSelector(lsvc, direction="backward", n_jobs=-1)
    sfs.fit(X_train_FS, y_train_FS)
    feature_idx = sfs.get_support()
    feature_name = X_train.columns[feature_idx]
    features_lSVC_sfs_backward.append(feature_name)

    forest = RandomForestClassifier(random_state=0, bootstrap=bootstrap_rf,  max_depth=max_depth_rf, max_features=max_features_rf, 
                                    min_samples_leaf=min_samples_leaf_rf, min_samples_split=min_samples_split_rf, n_estimators=n_estimators_rf)
    sfs = SequentialFeatureSelector(forest, direction="backward", n_jobs=-1)
    sfs.fit(X_train_FS, y_train_FS)
    feature_idx = sfs.get_support()
    feature_name = X_train.columns[feature_idx]
    features_RF_sfs_backward.append(feature_name)

    LogReg = LogisticRegression(C=C_LogReg,  penalty=penalty_LogReg, solver= solver_LogReg, tol=tol_LogReg, max_iter=10000,random_state=0)
    sfs = SequentialFeatureSelector(LogReg, direction="backward", n_jobs=-1)
    sfs.fit(X_train_FS, y_train_FS)
    feature_idx = sfs.get_support()
    feature_name = X_train.columns[feature_idx]
    features_LogReg_sfs_backward.append(feature_name)


features_lSVC_sfs_backward_all = []
for i in range(len(features_lSVC_sfs_backward)):
    for j in range(len(features_lSVC_sfs_backward[i])):
        features_lSVC_sfs_backward_all.append(features_lSVC_sfs_backward[i][j])
features_lSVC_sfs_backward_all = pd.DataFrame({"Counts":  pd.DataFrame(features_lSVC_sfs_backward_all).value_counts()})
features_lSVC_sfs_backward_all.reset_index()
ABCres = ABC_analysis(features_lSVC_sfs_backward_all.iloc[:,0])
ABCres["Aind"].reset_index(inplace=True)
feature_table.loc[ABCres["Aind"].iloc[:,0].tolist(), "features_lSVC_sfs_backward"] = 1

features_RF_sfs_backward_all = []
for i in range(len(features_RF_sfs_backward)):
    for j in range(len(features_RF_sfs_backward[i])):
        features_RF_sfs_backward_all.append(features_RF_sfs_backward[i][j])
features_RF_sfs_backward_all = pd.DataFrame({"Counts":  pd.DataFrame(features_RF_sfs_backward_all).value_counts()})
features_RF_sfs_backward_all.reset_index()
ABCres = ABC_analysis(features_RF_sfs_backward_all.iloc[:,0])
ABCres["Aind"].reset_index(inplace=True)
feature_table.loc[ABCres["Aind"].iloc[:,0].tolist(), "features_RF_sfs_backward"] = 1

features_LogReg_sfs_backward_all = []
for i in range(len(features_LogReg_sfs_backward)):
    for j in range(len(features_LogReg_sfs_backward[i])):
        features_LogReg_sfs_backward_all.append(
            features_LogReg_sfs_backward[i][j])
features_LogReg_sfs_backward_all = pd.DataFrame({"Counts":  pd.DataFrame(features_LogReg_sfs_backward_all).value_counts()})
features_LogReg_sfs_backward_all.reset_index()
ABCres = ABC_analysis(features_LogReg_sfs_backward_all.iloc[:,0])
ABCres["Aind"].reset_index(inplace=True)
feature_table.loc[ABCres["Aind"].iloc[:,0].tolist(), "features_LogReg_sfs_backward"] = 1

features_sfs_backward = pd.concat({"SVM": features_lSVC_sfs_backward_all,
                                  "RF": features_RF_sfs_backward_all, "LogReg": features_LogReg_sfs_backward_all}, axis=1)
variablenames = []
for i in range(len(features_sfs_backward .index)):
    variablenames.append(features_sfs_backward .index[i][0])
features_sfs_backward["variable"] = (variablenames)
features_sfs_backward.set_index("variable", inplace=True)

fig, ax = plt.subplots(figsize=(18, 16))
ax = sns.barplot(x=features_sfs_backward.index.tolist(),
                 y=features_sfs_backward.iloc[:, 0])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

# %% Save results in table
feature_table_olfdiag = feature_table.copy()

file = "/home/joern/feature_table_olfdiag.csv"
feature_table_olfdiag.to_csv(file)

# %% feature selection sum score for ABC for selection of final feature set

FS_sumscore_OlfDiag = feature_table_olfdiag.sum(axis = 1)
FS_sumscore_OlfDiag.sort_values(ascending = False, inplace=True)

with sns.axes_style("darkgrid"):
    fig = plt.figure(figsize=(12, 14))
    gs0 = gridspec.GridSpec(2, 2, figure=fig, wspace=.1, hspace=0.4)

    ax2 = fig.add_subplot(gs0[0, 0])
    ax1 = fig.add_subplot(gs0[0, 1])
    axes = [ax2, ax1]
    for i, ax in enumerate(axes):
        annotate_axes(ax,  str(string.ascii_lowercase[i]) + ")")

    ABC_A_FS_sumscore = ABC_analysis(
        ax=ax1, data=FS_sumscore_OlfDiag, PlotIt=True)
    ABC_A_FS_sumscore_nested = ABC_analysis(ABC_A_FS_sumscore["Aind"]["value"])

    barcols = ["dodgerblue" if (i) < ABC_A_FS_sumscore["ABlimit"] else "blue" if i <
               ABC_A_FS_sumscore_nested["ABlimit"] else "blue" for i in FS_sumscore_OlfDiag]
    ax1.set_title("ABC plot")
    sns.barplot(ax=ax2, x=FS_sumscore_OlfDiag.index.tolist(),
                y=FS_sumscore_OlfDiag, palette=barcols, alpha=1)
    ax2.set_title("Number of selections by 17 different methods")
    ax2.set_xlabel(None)
    ax2.set_ylabel("Times selected")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)

with sns.axes_style("darkgrid"):
    fig = plt.figure(figsize=(18, 14))
    gs0 = gridspec.GridSpec(2, 3, figure=fig, wspace=.1, hspace=0.4)

    ax2 = fig.add_subplot(gs0[0, 0])
    ax1 = fig.add_subplot(gs0[0, 1])
    ax3 = fig.add_subplot(gs0[0, 2])
    axes = [ax2, ax1, ax3]
    for i, ax in enumerate(axes):
        annotate_axes(ax,  str(string.ascii_lowercase[i]) + ")")

    ABC_A_FS_sumscore = ABC_analysis(
        ax=ax1, data=FS_sumscore_OlfDiag, PlotIt=True)
    ABC_A_FS_sumscore_nested = ABC_analysis(ax=ax3, data = ABC_A_FS_sumscore["Aind"]["value"], PlotIt=True)

    barcols = ["dodgerblue" if (i) < ABC_A_FS_sumscore["ABlimit"] else "blue" if i <
               ABC_A_FS_sumscore_nested["ABlimit"] else "darkblue" for i in FS_sumscore_OlfDiag]
    ax1.set_title("ABC plot")
    sns.barplot(ax=ax2, x=FS_sumscore_OlfDiag.index.tolist(),
                y=FS_sumscore_OlfDiag, palette=barcols, alpha=1)
    ax2.set_title("Number of selections by 17 different methods")
    ax2.set_xlabel(None)
    ax2.set_ylabel("Times selected")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)

#%% Test of selected features whether they suffice to predict the separate validation data subset
# Balanced accuracy and ROC AUC

BA_lsvc_fullFeatureSet, BA_RF_fullFeatureSet, BA_LogReg_fullFeatureSet = [], [], []
BA_lsvc_reducedFeatureSet, BA_RF_reducedFeatureSet, BA_LogReg_reducedFeatureSet = [], [], []
BA_lsvc_sparseFeatureSet, BA_RF_sparseFeatureSet, BA_LogReg_sparseFeatureSet = [], [], []
ROC_lsvc_fullFeatureSet, ROC_RF_fullFeatureSet, ROC_LogReg_fullFeatureSet = [], [], []
ROC_lsvc_reducedFeatureSet, ROC_RF_reducedFeatureSet, ROC_LogReg_reducedFeatureSet = [], [], []
ROC_lsvc_sparseFeatureSet, ROC_RF_sparseFeatureSet, ROC_LogReg_sparseFeatureSet = [], [], []

rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)
for train_index, test_index in rskf.split(X_train, y_train):
    X_train_FS, X_test_FS = X_train.iloc[train_index,
                                         :], X_train.iloc[test_index, :]
    y_train_FS, y_test_FS = y_train.iloc[train_index], y_train.iloc[test_index]
    
    X_train_validation, X_test_validation, y_train_validation, y_test_validation = train_test_split(
        X_test, y_test, test_size=0.8)

    # Full feature set
    lsvc = LinearSVC(C = C_lsvm, penalty = penalty_SVM, dual = dual_svm, loss = loss_svm, tol = tol_svm, max_iter=10000)
    lsvc.fit(X_train_FS, y_train_FS)
    y_pred = lsvc.predict(X_test_validation)
    BA_lsvc_fullFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    ROC_lsvc_fullFeatureSet.append(roc_auc_score(y_test_validation, y_pred))

    forest = RandomForestClassifier(random_state=0, bootstrap=bootstrap_rf,  max_depth=max_depth_rf, max_features=max_features_rf, 
                                    min_samples_leaf=min_samples_leaf_rf, min_samples_split=min_samples_split_rf, n_estimators=n_estimators_rf)
    forest.fit(X_train_FS, y_train_FS)
    y_pred = forest.predict(X_test_validation)
    BA_RF_fullFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    ROC_RF_fullFeatureSet.append(roc_auc_score(y_test_validation, y_pred))

    LogReg = LogisticRegression(C=C_LogReg,  penalty=penalty_LogReg, solver= solver_LogReg, tol=tol_LogReg, max_iter=10000,random_state=0)
    LogReg.fit(X_train_FS, y_train_FS)
    y_pred = LogReg.predict(X_test_validation)
    BA_LogReg_fullFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    ROC_LogReg_fullFeatureSet.append(roc_auc_score(y_test_validation, y_pred))


    # Reduced feature set
    X_train_FS_reduced = X_train_FS[["Dis", "log Thr", "Id", "log Distance left nostril", "correct_Discrimination_task", "Age", "ImportofO_Evaluation"]]
    X_test_validation_reduced = X_test_validation[["Dis", "log Thr", "Id", "log Distance left nostril", "correct_Discrimination_task", "Age", "ImportofO_Evaluation"]]
    
    lsvc = LinearSVC(C = C_lsvm, penalty = penalty_SVM, dual = dual_svm, loss = loss_svm, tol = tol_svm, max_iter=10000)
    lsvc.fit(X_train_FS_reduced, y_train_FS)
    y_pred = lsvc.predict(X_test_validation_reduced)
    BA_lsvc_reducedFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    ROC_lsvc_reducedFeatureSet.append(roc_auc_score(y_test_validation, y_pred))

    forest = RandomForestClassifier(random_state=0, bootstrap=bootstrap_rf,  max_depth=max_depth_rf, max_features=max_features_rf, 
                                    min_samples_leaf=min_samples_leaf_rf, min_samples_split=min_samples_split_rf, n_estimators=n_estimators_rf)
    forest.fit(X_train_FS_reduced, y_train_FS)
    y_pred = forest.predict(X_test_validation_reduced)
    BA_RF_reducedFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    ROC_RF_reducedFeatureSet.append(roc_auc_score(y_test_validation, y_pred))

    LogReg = LogisticRegression(C=C_LogReg,  penalty=penalty_LogReg, solver= solver_LogReg, tol=tol_LogReg, max_iter=10000,random_state=0)
    LogReg.fit(X_train_FS_reduced, y_train_FS)
    y_pred = LogReg.predict(X_test_validation_reduced)
    BA_LogReg_reducedFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    ROC_LogReg_reducedFeatureSet.append(roc_auc_score(y_test_validation, y_pred))
    
    # Sparse feature set
    X_train_FS_sparse = X_train_FS[["Dis", "log Thr", "Id"]]
    X_test_validation_sparse = X_test_validation[["Dis", "log Thr", "Id"]]
    
    lsvc = LinearSVC(C = C_lsvm, penalty = penalty_SVM, dual = dual_svm, loss = loss_svm, tol = tol_svm, max_iter=10000)
    lsvc.fit(X_train_FS_sparse, y_train_FS)
    y_pred = lsvc.predict(X_test_validation_sparse)
    BA_lsvc_sparseFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    ROC_lsvc_sparseFeatureSet.append(roc_auc_score(y_test_validation, y_pred))

    forest = RandomForestClassifier(random_state=0, bootstrap=bootstrap_rf,  max_depth=max_depth_rf, max_features=max_features_rf, 
                                    min_samples_leaf=min_samples_leaf_rf, min_samples_split=min_samples_split_rf, n_estimators=n_estimators_rf)
    forest.fit(X_train_FS_sparse, y_train_FS)
    y_pred = forest.predict(X_test_validation_sparse)
    BA_RF_sparseFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    ROC_RF_sparseFeatureSet.append(roc_auc_score(y_test_validation, y_pred))

    LogReg = LogisticRegression(C=C_LogReg,  penalty=penalty_LogReg, solver= solver_LogReg, tol=tol_LogReg, max_iter=10000,random_state=0)
    LogReg.fit(X_train_FS_sparse, y_train_FS)
    y_pred = LogReg.predict(X_test_validation_sparse)
    BA_LogReg_sparseFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    ROC_LogReg_sparseFeatureSet.append(roc_auc_score(y_test_validation, y_pred))


CV_results_BA = pd.DataFrame(np.column_stack([BA_lsvc_fullFeatureSet, BA_RF_fullFeatureSet, BA_LogReg_fullFeatureSet, 
                                           BA_lsvc_reducedFeatureSet, BA_RF_reducedFeatureSet, BA_LogReg_reducedFeatureSet, 
                                           BA_lsvc_sparseFeatureSet, BA_RF_sparseFeatureSet, BA_LogReg_sparseFeatureSet]),
                          columns = ["BA_lsvc_fullFeatureSet", "BA_RF_fullFeatureSet", "BA_LogReg_fullFeatureSet", 
                                                                     "BA_lsvc_reducedFeatureSet", "BA_RF_reducedFeatureSet", "BA_LogReg_reducedFeatureSet", 
                                                                     "BA_lsvc_sparseFeatureSet", "BA_RF_sparseFeatureSet", "BA_LogReg_sparseFeatureSet"])

CV_results_BA.mean()
CV_results_BA.std()
CV_results_BA.quantile()
CV_results_BA.quantile(0.025)
CV_results_BA.quantile(0.975)

ax = sns.boxplot(data = CV_results_BA)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

CV_results_ROC = pd.DataFrame(np.column_stack([ROC_lsvc_fullFeatureSet, ROC_RF_fullFeatureSet, ROC_LogReg_fullFeatureSet, 
                                           ROC_lsvc_reducedFeatureSet, ROC_RF_reducedFeatureSet, ROC_LogReg_reducedFeatureSet, 
                                           ROC_lsvc_sparseFeatureSet, ROC_RF_sparseFeatureSet, ROC_LogReg_sparseFeatureSet]),
                          columns = ["ROC_lsvc_fullFeatureSet", "ROC_RF_fullFeatureSet", "ROC_LogReg_fullFeatureSet", 
                                                                     "ROC_lsvc_reducedFeatureSet", "ROC_RF_reducedFeatureSet", "ROC_LogReg_reducedFeatureSet", 
                                                                     "ROC_lsvc_sparseFeatureSet", "ROC_RF_sparseFeatureSet", "ROC_LogReg_sparseFeatureSet"])

CV_results_ROC.mean()
CV_results_ROC.std()
CV_results_ROC.quantile()
CV_results_ROC.quantile(0.025)
CV_results_ROC.quantile(0.975)

fig, ax = plt.subplots(figsize=(18, 16))
ax = sns.boxplot(data = CV_results_ROC)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

# %% OlfDiag explanation with cluster features

# Balanced accuracy and ROC AUC

BA_lsvc_fullFeatureSet, BA_RF_fullFeatureSet, BA_LogReg_fullFeatureSet = [], [], []
BA_lsvc_reducedFeatureSet, BA_RF_reducedFeatureSet, BA_LogReg_reducedFeatureSet = [], [], []
BA_lsvc_sparseFeatureSet, BA_RF_sparseFeatureSet, BA_LogReg_sparseFeatureSet = [], [], []
ROC_lsvc_fullFeatureSet, ROC_RF_fullFeatureSet, ROC_LogReg_fullFeatureSet = [], [], []
ROC_lsvc_reducedFeatureSet, ROC_RF_reducedFeatureSet, ROC_LogReg_reducedFeatureSet = [], [], []
ROC_lsvc_sparseFeatureSet, ROC_RF_sparseFeatureSet, ROC_LogReg_sparseFeatureSet = [], [], []

rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)
for train_index, test_index in rskf.split(X_train, y_train):
    X_train_FS, X_test_FS = X_train.iloc[train_index,
                                         :], X_train.iloc[test_index, :]
    y_train_FS, y_test_FS = y_train.iloc[train_index], y_train.iloc[test_index]
    
    X_train_validation, X_test_validation, y_train_validation, y_test_validation = train_test_split(
        X_test, y_test, test_size=0.8)

    # Full feature set
    lsvc = LinearSVC(C = C_lsvm, penalty = penalty_SVM, dual = dual_svm, loss = loss_svm, tol = tol_svm, max_iter=10000)
    lsvc.fit(X_train_FS, y_train_FS)
    y_pred = lsvc.predict(X_test_validation)
    BA_lsvc_fullFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    ROC_lsvc_fullFeatureSet.append(roc_auc_score(y_test_validation, y_pred))

    forest = RandomForestClassifier(random_state=0, bootstrap=bootstrap_rf,  max_depth=max_depth_rf, max_features=max_features_rf, 
                                    min_samples_leaf=min_samples_leaf_rf, min_samples_split=min_samples_split_rf, n_estimators=n_estimators_rf)
    forest.fit(X_train_FS, y_train_FS)
    y_pred = forest.predict(X_test_validation)
    BA_RF_fullFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    ROC_RF_fullFeatureSet.append(roc_auc_score(y_test_validation, y_pred))

    LogReg = LogisticRegression(C=C_LogReg,  penalty=penalty_LogReg, solver= solver_LogReg, tol=tol_LogReg, max_iter=10000,random_state=0)
    LogReg.fit(X_train_FS, y_train_FS)
    y_pred = LogReg.predict(X_test_validation)
    BA_LogReg_fullFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    ROC_LogReg_fullFeatureSet.append(roc_auc_score(y_test_validation, y_pred))


    # Reduced feature set
    X_train_FS_reduced = X_train_FS[["Dis", "log Thr", "Id", "log Distance left nostril", "correct_Discrimination_task", "Age", "ImportofO_Evaluation"]]
    X_test_validation_reduced = X_test_validation[["Dis", "log Thr", "Id", "log Distance left nostril", "correct_Discrimination_task", "Age", "ImportofO_Evaluation"]]
    
    lsvc = LinearSVC(C = C_lsvm, penalty = penalty_SVM, dual = dual_svm, loss = loss_svm, tol = tol_svm, max_iter=10000)
    lsvc.fit(X_train_FS_reduced, y_train_FS)
    y_pred = lsvc.predict(X_test_validation_reduced)
    BA_lsvc_reducedFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    ROC_lsvc_reducedFeatureSet.append(roc_auc_score(y_test_validation, y_pred))

    forest = RandomForestClassifier(random_state=0, bootstrap=bootstrap_rf,  max_depth=max_depth_rf, max_features=max_features_rf, 
                                    min_samples_leaf=min_samples_leaf_rf, min_samples_split=min_samples_split_rf, n_estimators=n_estimators_rf)
    forest.fit(X_train_FS_reduced, y_train_FS)
    y_pred = forest.predict(X_test_validation_reduced)
    BA_RF_reducedFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    ROC_RF_reducedFeatureSet.append(roc_auc_score(y_test_validation, y_pred))

    LogReg = LogisticRegression(C=C_LogReg,  penalty=penalty_LogReg, solver= solver_LogReg, tol=tol_LogReg, max_iter=10000,random_state=0)
    LogReg.fit(X_train_FS_reduced, y_train_FS)
    y_pred = LogReg.predict(X_test_validation_reduced)
    BA_LogReg_reducedFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    ROC_LogReg_reducedFeatureSet.append(roc_auc_score(y_test_validation, y_pred))
    
    # Sparse feature set
    X_train_FS_sparse = X_train_FS[["errordistances_EUG", "errordistances_PEA"]]
    X_test_validation_sparse = X_test_validation[["errordistances_EUG", "errordistances_PEA"]]
    
    lsvc = LinearSVC(C = C_lsvm, penalty = penalty_SVM, dual = dual_svm, loss = loss_svm, tol = tol_svm, max_iter=10000)
    lsvc.fit(X_train_FS_sparse, y_train_FS)
    y_pred = lsvc.predict(X_test_validation_sparse)
    BA_lsvc_sparseFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    ROC_lsvc_sparseFeatureSet.append(roc_auc_score(y_test_validation, y_pred))

    forest = RandomForestClassifier(random_state=0, bootstrap=bootstrap_rf,  max_depth=max_depth_rf, max_features=max_features_rf, 
                                    min_samples_leaf=min_samples_leaf_rf, min_samples_split=min_samples_split_rf, n_estimators=n_estimators_rf)
    forest.fit(X_train_FS_sparse, y_train_FS)
    y_pred = forest.predict(X_test_validation_sparse)
    BA_RF_sparseFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    ROC_RF_sparseFeatureSet.append(roc_auc_score(y_test_validation, y_pred))

    LogReg = LogisticRegression(C=C_LogReg,  penalty=penalty_LogReg, solver= solver_LogReg, tol=tol_LogReg, max_iter=10000,random_state=0)
    LogReg.fit(X_train_FS_sparse, y_train_FS)
    y_pred = LogReg.predict(X_test_validation_sparse)
    BA_LogReg_sparseFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    ROC_LogReg_sparseFeatureSet.append(roc_auc_score(y_test_validation, y_pred))


CV_results_BA = pd.DataFrame(np.column_stack([BA_lsvc_fullFeatureSet, BA_RF_fullFeatureSet, BA_LogReg_fullFeatureSet, 
                                           BA_lsvc_reducedFeatureSet, BA_RF_reducedFeatureSet, BA_LogReg_reducedFeatureSet, 
                                           BA_lsvc_sparseFeatureSet, BA_RF_sparseFeatureSet, BA_LogReg_sparseFeatureSet]),
                          columns = ["BA_lsvc_fullFeatureSet", "BA_RF_fullFeatureSet", "BA_LogReg_fullFeatureSet", 
                                                                     "BA_lsvc_reducedFeatureSet", "BA_RF_reducedFeatureSet", "BA_LogReg_reducedFeatureSet", 
                                                                     "BA_lsvc_sparseFeatureSet", "BA_RF_sparseFeatureSet", "BA_LogReg_sparseFeatureSet"])

CV_results_BA.mean()
CV_results_BA.std()
CV_results_BA.quantile()
CV_results_BA.quantile(0.025)
CV_results_BA.quantile(0.975)

fig, ax = plt.subplots(figsize=(18, 16))
ax = sns.boxplot(data = CV_results_BA)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

CV_results_ROC = pd.DataFrame(np.column_stack([ROC_lsvc_fullFeatureSet, ROC_RF_fullFeatureSet, ROC_LogReg_fullFeatureSet, 
                                           ROC_lsvc_reducedFeatureSet, ROC_RF_reducedFeatureSet, ROC_LogReg_reducedFeatureSet, 
                                           ROC_lsvc_sparseFeatureSet, ROC_RF_sparseFeatureSet, ROC_LogReg_sparseFeatureSet]),
                          columns = ["ROC_lsvc_fullFeatureSet", "ROC_RF_fullFeatureSet", "ROC_LogReg_fullFeatureSet", 
                                                                     "ROC_lsvc_reducedFeatureSet", "ROC_RF_reducedFeatureSet", "ROC_LogReg_reducedFeatureSet", 
                                                                     "ROC_lsvc_sparseFeatureSet", "ROC_RF_sparseFeatureSet", "ROC_LogReg_sparseFeatureSet"])

CV_results_ROC.mean()
CV_results_ROC.std()
CV_results_ROC.quantile()
CV_results_ROC.quantile(0.025)
CV_results_ROC.quantile(0.975)

fig, ax = plt.subplots(figsize=(18, 16))
ax = sns.boxplot(data = CV_results_ROC)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

# %% Cluster versus OlfDiag

FS_sumscore_diff =  FS_sumscore_cluster - FS_sumscore_OlfDiag
dfFS_sumscore_diff = pd.DataFrame(FS_sumscore_diff.index.tolist(), columns = ["variable"])
dfFS_sumscore_diff["diff"] = list(FS_sumscore_diff)
dfFS_sumscore_diff.sort_values("diff", ascending = False, inplace=True)

#dfFS_sumscore_diff.reset_index(inplace=True)
dfFS_sumscore_diff.set_index("variable",inplace=True) 
#dfFS_sumscore_diff.reset_index(inplace=True)
    
with sns.axes_style("darkgrid"):
    fig, ax1 = plt.subplots(figsize=(12, 7))

    barcols = ["dodgerblue" if (i) < 6 else "blue" for i in dfFS_sumscore_diff["diff"].to_numpy()]
    sns.barplot(ax = ax1, x=dfFS_sumscore_diff.index.tolist(),
                y=dfFS_sumscore_diff["diff"], palette = barcols, alpha=1)
    ax1.set_title("Difference 'cluster - olfactory diagnosis' in the number of selections by 17 different methods")
    ax1.set_xlabel(None)
    ax1.set_ylabel("Delta times selected")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)