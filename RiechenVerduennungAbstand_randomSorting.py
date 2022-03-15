#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 19:18:53 2022

@author: joern
"""

import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt



nPermut = 1000
# # %% Sorting task distances
# Sortingresult = []
# for res in range(nPermut):
#     sort = np.zeros((1,5))
#     for i in range(1):
#         l = list(range(1,6))
#         #l = [5,4,3,2,1]
#         random.shuffle(l)
#         sort[i,:] = l
    
#         correct = [range(1,6)] 
#         dists = abs(sort - correct)
#         sumdists = dists.sum(axis = 1) 
#         sumdists = 12 - sumdists
#         Sortingresult.append(sumdists.sum())

# print(max(Sortingresult ))
# Sortingresult = np.array(Sortingresult) / 12

fig, ax = plt.subplots(figsize = (30, 10))
# sns.kdeplot(Sortingresult, label = "Sorting one distance", ax = ax, color = "purple")
# sns.histplot(Sortingresult,   stat="density", ax = ax, color = "purple")
# ax.set_xlim(0,1)

# sort = np.zeros((nPermut,5))
# for i in range(nPermut):
#     l = list(range(1,6))
#     #l = [5,4,3,2,1]
#     random.shuffle(l)
#     sort[i,:] = l

# correct = [range(1,6)] 
# dists = abs(sort - correct)
# sumdists = dists.sum(axis = 1) 
# sumdists = 12 - sumdists
# print(max(sumdists ))
# sumdists = sumdists / max(sumdists 
#
# sns.kdeplot(sumdists.flatten())
# sns.histplot(sumdists .flatten(),   stat="density")

# %% Sorting task distances two tests

Sortingresult = []
for res in range(nPermut):
    sort = np.zeros((2,5))
    for i in range(2):
        l = list(range(1,6))
        #l = [5,4,3,2,1]
        random.shuffle(l)
        sort[i,:] = l
    
        correct = [range(1,6)] 
        dists = abs(sort - correct)
        sumdists = dists.sum(axis = 1) 
        sumdists = 12 -  sumdists
        Sortingresult.append(sumdists.sum())

print(max(Sortingresult ))
Sortingresult = np.array(Sortingresult) / 24

#fig, ax = plt.subplots(figsize = (30, 10))
sns.kdeplot(Sortingresult, label = "Sorting twp distances", ax = ax, color = "red")
sns.histplot(Sortingresult,   stat="density", ax = ax, color = "red")
ax.set_xlim(0,1)

# # %% Sorting task only completely correct

# Sortingresult = []
# for res in range(nPermut):
#     sort = np.zeros((1,5))
#     for i in range(1):
#         l = list(range(1,6))
#         #l = [5,4,3,2,1]
#         random.shuffle(l)
#         sort[i,:] = l
    
#         correct = [range(1,6)] 
#         dists = abs(sort - correct)
#         sumhits = (dists == 0).sum() 
#         Sortingresult.append(sumhits)

# print(max(Sortingresult ))
# Sortingresult = np.array(Sortingresult) / 5

# #fig, ax = plt.subplots(figsize = (30, 10))
# sns.kdeplot(Sortingresult, label = "Sorting twp distances", ax = ax, color = "grey")
# sns.histplot(Sortingresult,   stat="density", ax = ax, color = "grey")
# ax.set_xlim(0,1)

# # %% Sorting task only completely correct two tests

# Sortingresult = []
# for res in range(nPermut):
#     sort = np.zeros((2,5))
#     for i in range(2):
#         l = list(range(1,6))
#         #l = [5,4,3,2,1]
#         random.shuffle(l)
#         sort[i,:] = l
    
#         correct = [range(1,6)] 
#         dists = abs(sort - correct)
#         sumhits = (dists == 0).sum() 
#         Sortingresult.append(sumhits)

# print(max(Sortingresult ))
# Sortingresult = np.array(Sortingresult) / 10

# #fig, ax = plt.subplots(figsize = (30, 10))
# sns.kdeplot(Sortingresult, label = "Sorting twp distances", ax = ax, color = "black")
# sns.histplot(Sortingresult,   stat="density", ax = ax, color = "black")
# ax.set_xlim(0,1)

# %% Odor identifictaion

IDresult = []
for res in range(nPermut):
    correct = np.zeros((16,4))
    for i in range(16):
        l = [1, 0, 0, 0]
        random.shuffle(l)
        correct[i,:] = l
    sumcorrect = 0
    for i in range(16):
        np.random.seed(res+i)
        sumcorrect += correct[i,np.random.randint(4)]
    IDresult.append(sumcorrect)

print(max(IDresult ))
IDresult = np.array(IDresult) / 16

#fig, ax = plt.subplots(figsize = (30, 10))
sns.kdeplot(IDresult, label = "Odor identifictaion", ax = ax, color = "green")
sns.histplot(IDresult,   stat="density", ax = ax, color = "green")
ax.set_xlim(0,1)

# %% Odor discrimination

Discrresult = []
for res in range(nPermut):
    correct = np.zeros((16,3))
    for i in range(16):
        l = [1, 0, 0]
        random.shuffle(l)
        correct[i,:] = l
    sumcorrect = 0
    for i in range(16):
        np.random.seed(res+i)
        sumcorrect += correct[i,np.random.randint(3)]
    Discrresult.append(sumcorrect)

print(max(Discrresult ))
Discrresult = np.array(Discrresult) / 16

#fig, ax = plt.subplots(figsize = (30, 10))
sns.kdeplot(Discrresult, label = "Odor discrimination", ax = ax, color = "yellow")
sns.histplot(Discrresult,   stat="density", ax = ax, color = "yellow")
ax.set_xlim(0,1)


#%% Odor threshold staircase
def ct1t2():
    t1 = [1, 0, 0]
    random.shuffle(t1)
    t2 = [1, 0, 0]
    random.shuffle(t2)
    t1t2 = t1[np.random.randint(3)] + t2[np.random.randint(3)]
    return(t1t2)

def up(Thr):
    while ct1t2() == 2:
        Thr += 1
    return(Thr)

def down(Thr):
    while ct1t2() < 2:
        Thr -= 1
    return(Thr)

Thresresult = []
for res in range(nPermut):
        
    turn = []
    Thr = 16
    for i in range(4):
        Thr = down(Thr)
        Thr = Thr if Thr >= 1 else 1
        turn.append(Thr)
        Thr = up(Thr)
        Thr += 1
        Thr = Thr if Thr <= 16 else 16
        turn.append(Thr)
    Thresresult.append(np.array(turn[3:7]).mean() )    

Thresresult = (np.array(Thresresult)  - 1) / 15


#fig, ax = plt.subplots(figsize = (30, 10))
sns.kdeplot(Thresresult, label = "Olfactory threshold", ax = ax, color = "brown")
sns.histplot(Thresresult,   stat="density", ax = ax, color = "brown")
ax.set_xlim(0,1)
