#!/usr/bin/env python
# coding: utf-8

# In[68]:


import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import os 
import urllib
import datetime
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Dropout


# In[85]:


class analysis(): 
    def __init__(self): 
        self.train = "dengue_features_train.csv"
        self.test = "dengue_features_test.csv"
        self.labels = "dengue_labels_train.csv"
        
    def load_data(self): 
        self.data = pd.read_csv(self.train)
        self.data["index"] = [i for i in range(self.data.shape[0])]
        self.test = pd.read_csv(self.test)
        print("Training dim: {}".format(self.data))
        print("Testing dim: {}".format(self.test))
        
    def find_clust_train(self):
        self.clust = self.data.iloc[:,8:]
        self.clust['index']=self.data['index']
        self.clust = self.clust.dropna()
        self.index = self.clust['index']
        self.clust = self.clust.drop(['index'],axis=1)
        scaler = MinMaxScaler(feature_range=(0,1))
        self.df = pd.DataFrame(scaler.fit_transform(self.clust))
        plt.style.use("dark_background")
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(self.df)
        colors = []
        for i in pcs: 
            if i[0]<0.05 and i[1]<0:
                colors.append("blue")
            if i[0]<0.05 and i[1]>0:
                colors.append("green")
            if i[0]>0.05 and i[1]<0:
                colors.append("red")
            if i[0]>0.05 and i[1]>0:
                colors.append("orange")
        
        f = plt.figure(figsize=(20,14))
        f.suptitle("PCA on non NaN features",fontsize=40)
        ax = f.add_subplot(121)
        ax1 = f.add_subplot(122)
        ax.scatter([i[0] for i in pcs],[i[1] for i in pcs])
        ax1.scatter([i[0] for i in pcs],[i[1] for i in pcs],c=colors)
        axs = {ax:"Raw",ax1:"Splitted"}
        for Ax in axs:
            Ax.set_xlabel("PC1",fontsize=25)
            Ax.set_ylabel("PC2",fontsize=25)
            Ax.set_title(axs[Ax],fontsize=35)
        plt.show()
        self.colors = colors
        
    def inpute_train_on_clust(self):
        self.data = self.data[self.data['index'].isin(self.index)]
        self.data["clust"] = self.colors
        
        self.C1 = self.data[self.data["clust"]=="blue"]
        self.C2 = self.data[self.data["clust"]=="green"]
        self.C3 = self.data[self.data["clust"]=="red"]
        self.C4 = self.data[self.data["clust"]=="orange"]
        
        self.C1.iloc[:,4:-1] = self.C1.iloc[:,4:-1].apply(lambda x: x.fillna(x.mean()),axis=0)
        self.C2.iloc[:,4:-1] = self.C2.iloc[:,4:-1].apply(lambda x: x.fillna(x.mean()),axis=0)
        self.C3.iloc[:,4:-1] = self.C3.iloc[:,4:-1].apply(lambda x: x.fillna(x.mean()),axis=0)
        self.C4.iloc[:,4:-1] = self.C4.iloc[:,4:-1].apply(lambda x: x.fillna(x.mean()),axis=0)
        self.data = pd.concat([self.C1,self.C2,self.C3,self.C4],axis=0)
        self.X_train = self.data.drop(["index","clust"],axis=1)
        print(self.data)
    
    def find_clust_test(self):
        self.index = [i for i in range(self.test.shape[0])]
        self.clust = self.test.iloc[:,6:19].fillna(0)
        scaler = MinMaxScaler(feature_range=(0,1))
        self.df = pd.DataFrame(scaler.fit_transform(self.clust))
        plt.style.use("dark_background")
        pca = PCA(n_components=10)
        pcs = pca.fit_transform(self.df)
        f = plt.figure(figsize=(20,14))
        f.suptitle("PCA on non NaN features",fontsize=40)
        ax = f.add_subplot(121)
        ax1 = f.add_subplot(122)
        kmeans = KMeans(n_clusters=3, random_state=0).fit(pcs)
        color = ["green","red","blue","yellow","orange"]
        dic = [color[i] for i in kmeans.labels_]
        ax.scatter([i[0] for i in pcs],[i[1] for i in pcs],)
        ax1.scatter([i[0] for i in pcs],[i[1] for i in pcs],c=dic)
        axs = {ax:"Raw",ax1:"Kmeans cluster"}
        for Ax in axs:
            Ax.set_xlabel("PC1",fontsize=25)
            Ax.set_ylabel("PC2",fontsize=25)
            Ax.set_title(axs[Ax],fontsize=35)
        plt.show()
        self.clust["clust"]=kmeans.labels_
        
    def impute_test_on_clust(self):
        self.C1 = self.test[self.clust["clust"]==0]
        self.C2 = self.test[self.clust["clust"]==1]
        self.C3 = self.test[self.clust["clust"]==2]
        
        self.C1.iloc[:,4:] = self.C1.iloc[:,4:].apply(lambda x: x.fillna(x.mean()),axis=0)
        self.C2.iloc[:,4:] = self.C2.iloc[:,4:].apply(lambda x: x.fillna(x.mean()),axis=0)
        self.C3.iloc[:,4:] = self.C3.iloc[:,4:].apply(lambda x: x.fillna(x.mean()),axis=0)
        self.X_test = pd.concat([self.C1,self.C2,self.C3],axis=0)

    def model(self):
        print(self.X_train.columns,self.X_test.columns)
        print(np.unique(self.X_train['city']))
        
o = analysis()
o.load_data()
o.find_clust_train()
o.inpute_train_on_clust()
o.find_clust_test()
o.impute_test_on_clust()
o.model()
# TO DO: 

