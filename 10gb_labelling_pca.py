#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Following code performes dimensionality reduction (PCA) for 
different number of components and trained GB model.
"""
import time
from PIL import Image
import os
import cv2
import glob
import lightgbm as lgb
import random
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import roc_auc_score
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.decomposition import PCA
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import random
random.seed(1)
def train_lgbm(train_x, train_y): 
    """
    returns trained lgbm model
    
    """
    num_round=70
    param = {
        "objective": "multiclass",
        "num_class": 2,
        "metric": ["multi_logloss", "multi_error"],
        "verbose": -1,
        "learning_rate": 0.1,
        "num_leaves": 191,
        "feature_fraction": 0.46,
        "bagging_fraction": 0.69,
        "bagging_freq": 0,
        "max_depth": 7,
    }
    X_train, X_val, y_train, y_val = train_test_split(
                    train_x, train_y, test_size=0.2,random_state=42)   # split into validation and training     
    #a=np.ones(len(y_train))/0.8;   #use this if you want to assign weights to each class
    #for i in range(0,len(y_train)):
    #    if train_y[i]==0:
    #        a[i]=1
    train_data=lgb.Dataset(X_train,y_train) #weight=a 
    vali_data=lgb.Dataset(X_val, y_val, reference=train_data)
    gbm = lgb.train(param, train_data, num_round, valid_sets=[vali_data],verbose_eval=-1 ) 
    print('model appended')
    return gbm   
def getpreds(models, x_test,y_test,w=None):
    """
    ---input---
    models- list of fitted models
    x_test- test instances
    y_test- test labels
    w- the weight of each classifier in models
    ---returns---
    final accuracy,
    labels: argmax from the weighted average of probabilities being 1 or 0
    """
    probabs=[]
    i=1
    for clf in models:
        pred=clf.predict(x_test)
        p=np.argmax(pred,axis=1)
        i+=1
        probabs.append(pred)       
    probabs=np.array(probabs)
    labels = np.average(probabs, axis=0, weights=w)
    labels=np.argmax(labels,axis=1)
    accuracy_test = accuracy_score(y_test,labels)*100
    return accuracy_test,labels
def unison_shuffled_copies(a, b,c=None):   
    assert len(a) == len(b)
    p = np.random.RandomState(seed=42).permutation(len(a))
    a=np.array([a[i] for i in p])
    b=np.array([b[i] for i in p])
    return a,b  
def makegroups(a,b,c=None):
    """
    ---inputs---
    a- features set
    b- labels
    ---returns--
    10 groups of dataset half of 
    the original size
    """
    xtrain10=[]
    ytrain10=[]
    for i in range (0,5): 
        a,b=unison_shuffled_copies(a,b)
        splitx=np.split(a,2)
        splity=np.split(b,2)
        xtrain10.append(np.array(splitx[0]))
        xtrain10.append(np.array(splitx[1]))
        ytrain10.append(np.array(splity[0]))
        ytrain10.append(np.array(splity[1]))
        print('makinggroupsdone')
    return xtrain10,ytrain10
def getlabels(source1,source2=None):
    """
    --inputs--
    source1,source2- path to the image folders
    --returns--
    extracts labels from the image name
    """
    labels=[]
    files= os.listdir(source1)  
    files2=os.listdir(source2)  
    for file in files:   
        file_name=os.path.splitext(os.path.basename(file))[0]
        label=int(file_name.split('_')[4][5])
        labels.append(label) 
    if source2!=None:
        for file in files2:   
            file_name=os.path.splitext(os.path.basename(file))[0]
            label=int(file_name.split('_')[4][5])
            labels.append(label)
    labels=np.array(labels)
    return labels

def getnpfeatures(source1,source2=None,source3=None,n=None): 
    """
    ---input--
    This function is used only on the original raw patches
    source1,source2,source paths to training/validation/synthetic image folders
    n-number of instances to take from each folder. none by default
    
    ---returns--
    returns numpy arrays of features
    """
    data=[]
    for myFile in glob.glob (source1):   #training data input dimension is resized from
        image = load_img(myFile, target_size=(50, 50))
        image = img_to_array(image) 
        data.append(image)
    if source2!=None:
        i=0
        for myFile in glob.glob (source2):
            if i!=n:
                image = load_img(myFile, target_size=(50, 50))
                image = img_to_array(image) 
                data.append(image)
                i+=1
    if source3!=None:
        i=0
        for myFile in glob.glob (source3):
            if i!=n:
                image = load_img(myFile, target_size=(50, 50))
                image = img_to_array(image) 
                data.append(image)   
                i+=1
    data=np.array(data)
    return data
if __name__=="__main__":    
    train_x=getnpfeatures('/home/ubuntu/preprocessing/maincode/files/subset_training/*.png')  #the set that is assumed to be labeled
    train_y=getlabels('/home/ubuntu/preprocessing/maincode/files/subset_training/')
    train_x, test_x, train_y, test_y = train_test_split(
                    train_x, train_y, test_size=0.90,random_state=42)                         
    train_x = (train_x - np.min(train_x, 0)) / (np.max(train_x, 0) + 0.0001)   #rescale to 0/1
    test_x = (test_x - np.min(test_x, 0)) / (np.max(test_x, 0) + 0.0001) 
    test_x=np.reshape(test_x,(-1,7500))  #1408
    train_x=np.reshape(train_x,(-1,7500))
    comps=[50,100,200,250]  #for each number of principal components ...
    for i in comps: 
        print('when number of components is ',i)
        pca = PCA(n_components=i)
        pca.fit(train_x)
        pca_train_x= pca.transform(train_x)
        pca_test_x = pca.transform(test_x)
        trx,tryy=makegroups(pca_train_x,train_y)  
        trx=np.array(trx)
        tryy=np.array(tryy) 
        accs=0
        aucs=0
        best_acc=0
        f1=0
        rc=0
        pr=0
        start = time.time()
        it=1 
        for j in range(0,5):
            models=[]
            for x,y in zip(trx,tryy):  
                model=train_lgbm(x,y)
                models.append(model) 
            acc,labels=getpreds(models,pca_test_x, test_y) 
            print('accuracy for the current round', acc)
            auc=roc_auc_score(test_y,labels)
            print('balanced accuracy for the current round',auc)
            accs+=acc 
            aucs+=auc
            f1+=f1_score(test_y,labels)
            pr+=precision_score(test_y,labels)
            rc+=recall_score(test_y,labels)
        stop = time.time()
        duration = stop-start
        print('balanced accuracy over runs',aucs/5)
        print('labeling accuracy over 10 models averaged over 10 runs', accs/5)
        print('precision', pr/5)
        print('recall', rc/5)
        print('fmeasure', f1/5)
        print('confusion matrix', confusion_matrix(test_y,labels))
        #with open("/home/ubuntu/preprocessing/maincode/files/data/pca", "wb") as f:
        #    pickle.dump(labels, f) 
           


                 
