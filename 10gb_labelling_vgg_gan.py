#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
the vgg-16 features should be already extracted from training set and from synthetic images generated by GAN.
Desired number of synthetic images added to the training set can be specified.
Finally, it trains 10gb models and generates final labels for the rest of the 90% instances
"""
import time
from PIL import Image
import scipy.misc
import os
import lightgbm as lgb
import cv2
import glob
import random
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import roc_auc_score
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import random
random.seed(1)
def train_lgbm(train_x, train_y): 
    """
    returns lgbm model
    
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

def get_features_gan(x1,x2=None,x3=None,n=None):
    """
    used when applying VGG+GAN
    ---input---
    full paths of x1,x2,x3 VGG-feature vectors
    x1 is a path to VGG-features generated from inistial training set,
    x2,x3 are the paths to VGG-features generated from GAN
    n specifies number of synthetic images you want to add in the training set
    ---returns---
    np.array of features
    gets VGG features from the paths specified
    in that case get's n features from the second and third paths(GAN images)
    """
    data=[]
    for myFile in glob.glob (x1):   
        img_array= np.load(myFile) 
        data.append(img_array)
    if x2!=None:
        i=0
        for myFile in glob.glob (x2):   
            if i!=n:
                img_array= np.load(myFile) 
                data.append(img_array) 
                i+=1
    if x3!=None:
        i=0
        for myFile in glob.glob (x3):   
            if i!=n:
                img_array= np.load(myFile) 
                data.append(img_array) 
                i+=1           
    return np.array(data)

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
if __name__=="__main__":    
    n=6000 # number of synthetic images you want to add in your data per class.
    train_y=getlabels('/home/ubuntu/preprocessing/maincode/files/subset_features/VGG-1-224/')  
    train_x=get_features_gan('/home/ubuntu/preprocessing/maincode/files/subset_features/VGG-1-224/*.npy','/home/ubuntu/preprocessing/maincode/files/features_over_epochs/1_fakes_1200epochs/VGG-1-224/*.npy','/home/ubuntu/preprocessing/maincode/files/features_over_epochs/0_fakes_1200epochs/VGG-1-224/*.npy',n)    
    test_x=get_features('/home/ubuntu/preprocessing/maincode/files/training_features/VGG-1-224/*.npy','/home/ubuntu/preprocessing/maincode/files/validation_features/VGG-1-224/*.npy')
    test_y=getlabels('/home/ubuntu/preprocessing/maincode/files/training_features/VGG-1-224/','/home/ubuntu/preprocessing/maincode/files/validation_features/VGG-1-224/')   
    train_y_2=np.ones(n) 
    train_y=np.append(train_y,train_y_2)   
    train_y_3=np.zeros(n)
    train_y=np.append(train_y,train_y_3) 
    train_x=np.reshape(train_x,(-1,1408))  #1408   
    test_x=np.reshape(test_x,(-1,1408))   #1408      
    print('train dims',train_x.shape,'  ', train_y.shape ) 
    print('test dims',test_x.shape,'  ', test_y.shape )
    trx,tryy=makegroups(train_x,train_y)  
    trx=np.array(trx) # 10x500x50x50x3
    tryy=np.array(tryy) #10x500  
    accs=0
    aucs=0
    best_acc=0
    f1=0
    rc=0
    pr=0
    it=1
    start = time.time()
    for j in range(0,it):
        models=[]
        for x,y in zip(trx,tryy):
            model=train_lgbm(x,y)
            models.append(model) 
        acc,labels=getpreds(models,test_x, test_y) 
        auc=roc_auc_score(test_y,labels)
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
    print('confusion matrix', confusion_matrix(test_y,labels))    
    #with open("/home/ubuntu/preprocessing/maincode/files/masterthesis/labels/vgg_gan", "wb") as f:
    #    pickle.dump(labels, f) 
    

                 
