#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trains 1 GB model on the VGG features extracted 
from the training set

"""
import numpy as np
from PIL import Image
import scipy.misc
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import cv2
import glob
from keras import optimizers
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import lightgbm as lgb
import argparse
import time
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
random.seed(1)
def get_features(x1,x2=None,x3=None):
    """
    --input---
    takes x1,x2 as paths to VGG
    -- returns--
    VGG features in np.arrays
    """
    data=[]
    for myFile in glob.glob (x1):   
        img_array= np.load(myFile) 
        data.append(img_array)
    if x2!=None:
        for myFile in glob.glob (x2):   
            img_array= np.load(myFile) 
            data.append(img_array) 
    if x3!=None:
        for myFile in glob.glob (x3):   
            img_array= np.load(myFile) 
            data.append(img_array) 
    return np.array(data)  
def getlabels(source1,source2=None): #each image_name contains the label as a last symbol. This function takes the destination
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
if __name__=="__main__":
    print('1 grad labeling')
    train_y=getlabels('/home/ubuntu/preprocessing/maincode/files/subset_features/VGG-1-224/')  
    train_x=get_features('/home/ubuntu/preprocessing/maincode/files/subset_features/VGG-1-224/*.npy')
    train_x, test_x, train_y, test_y = train_test_split(
                    train_x, train_y, test_size=0.90,random_state=42)     
    train_x, val_x, train_y, val_y = train_test_split(
                        train_x, train_y, test_size=0.20,random_state=42)   # split into validation and traini
    train_x = (train_x - np.min(train_x, 0)) / (np.max(train_x, 0) + 0.0001)   #rescale to 0/1
    test_x = (test_x - np.min(test_x, 0)) / (np.max(test_x, 0) + 0.0001) 
    val_x = (val_x - np.min(val_x, 0)) / (np.max(val_x, 0) + 0.0001) 
    test_x=np.reshape(test_x,(-1,1408))  #1408
    train_x=np.reshape(train_x,(-1,1408))
    val_x=np.reshape(val_x,(-1,1408))
    print('test dims',test_x.shape,'  ', test_y.shape )
    print('train dims',train_x.shape,'  ', train_y.shape ) 
    print('val dims',val_x.shape,'  ', val_y.shape ) 
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
    accs=0
    aucs=0
    f1=0
    rc=0
    pr=0
    start = time.time()
    it=1
    print('training start')
    for j in range(0,it):
        train_data=lgb.Dataset(train_x, train_y)
        vali_data=lgb.Dataset(val_x, val_y, reference=train_data)
        gbm = lgb.train(param, train_data, num_round, valid_sets=[vali_data],verbose_eval=-1 ) 
        scores=gbm.predict(test_x)
        labels=np.argmax(scores, axis=1)
        acc = accuracy_score(labels,test_y)*100
        auc=roc_auc_score(test_y,labels)
        print('  balanced accuracy for the current round',auc)
        accs+=acc 
        aucs+=auc
        f1+=f1_score(test_y,labels)
        pr+=precision_score(test_y,labels)
        rc+=recall_score(test_y,labels)
    stop = time.time()
    duration = stop-start
    print('balanced accuracy over runs',aucs/it)
    print('labeling accuracy over 10 models averaged over 10 runs', accs/it)
    print('precision', pr/it)
    print('recall', rc/it)
    print('fmeasure', f1/it)

    