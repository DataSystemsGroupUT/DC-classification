#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code performs different labeling techniques
to find the best model for data labelling
"""
from PIL import Image
import numpy as np
import scipy.misc
import os
from sklearn.model_selection import cross_validate
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import cv2
import glob
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import argparse
import time
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
random.seed(1)
def get_features(x1,x2=None,x3=None):
    """
    --input---
    takes x1,x2 as paths to VGG features
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
    y_train=getlabels('/home/ubuntu/preprocessing/maincode/files/subset_features/VGG-1-224/')  
    X_train=get_features('/home/ubuntu/preprocessing/maincode/files/subset_features/VGG-1-224/*.npy') #,'/home/ubuntu/preprocessing/maincode/files/1_fakes_features/VGG-1-224/*.npy','/home/ubuntu/preprocessing/maincode/files/0_fakes_features/VGG-1-224/*.npy')    )   
    X_train, X_test, y_train, y_test = train_test_split(
                    X_train, y_train, test_size=0.90,random_state=42) # splits train:test in proportion of 1:9
    X_train=np.reshape(X_train,(-1,1408))  #1408 is the dimension of VGG feature vector
    X_test=np.reshape(X_test,(-1,1408))
    print('test dims',X_test.shape,'  ', y_test.shape )
    print('train dims',X_train.shape,'  ', y_train.shape )   
    start = time.time()
    my_models=[
    RandomForestClassifier(n_estimators=40,
                 max_depth=11,
                  min_samples_split=62,
                  min_samples_leaf=52,
                   bootstrap=True),
    DecisionTreeClassifier(),
    GaussianNB(),
    AdaBoostClassifier(),
    ]
    for m in my_models:
        print("="*30)
        name = m.__class__.__name__
        print(name)
        accs=0
        aucs=0
        f1=0
        rc=0
        pr=0
        it=30
        for j in range(0,it):
            cross_validate(m, X_train, y_train, 
                           cv=5)
            mod=m.fit(X_train,y_train)
            scores=mod.predict_proba(X_test)
            labels=np.argmax(scores, axis=1)
            acc = accuracy_score(labels,y_test)*100
            accs+=acc 
            auc=roc_auc_score(y_test,labels)
            print('balanced accuracy for the current round',auc)
            aucs+=auc
            f1+=f1_score(y_test,labels)
            pr+=precision_score(y_test,labels)
            rc+=recall_score(y_test,labels)
        stop = time.time()
        duration = stop-start
        print('balanced accuracy over runs',aucs/it)
        print('labeling accuracy over 10 models averaged over 10 runs', accs/it)
        print('precision', pr/it)
        print('recall', rc/it)
        print('fmeasure', f1/it)