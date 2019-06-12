"""
this script trains the best architecture found by nni on the full dataset
Option 1: uses the weights provided to reproduce the results, when all the original data is used
option 2: if you wish to train the same model but on the smaller set of labeled instances(NB! that is assumed to be labeled, can't be more than 12000)

"""
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
import pickle
from keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn import metrics
from keras.utils import plot_model
from sklearn.metrics import accuracy_score
import glob
from sklearn.utils import class_weight
from keras.models import model_from_json
from keras.utils import multi_gpu_model, to_categorical
from keras.optimizers import SGD
import os
import nni
import time
from nni.networkmorphism_tuner.graph import json_to_graph
from keras.callbacks import EarlyStopping, TensorBoard
from PIL import Image
from sklearn.metrics import roc_auc_score
import random
import keras.backend as K
import tensorflow as tf
import cv2
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

random.seed(1)
def build_graph_from_json(ir_model_json):
    """build model from json representation
    """
    graph = json_to_graph(ir_model_json)
    model = graph.produce_keras_model()
    return model
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
def auc(y_true, y_pred):
    """
    calculated area under the curve
    
    """
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc
def getnpfeatures(source1,source2=None,source3=None): 
    """
    ---input--
    This function is used only on the original raw patches
    source1,source2,source paths to training/validation image folders    
    ---returns--
    returns numpy arrays of features
    """
    data=[]
    for myFile in glob.glob (source1):   
        image = load_img(myFile, target_size=(32, 32))
        image = img_to_array(image) 
        data.append(image)
    if source2!=None:
        for myFile in glob.glob (source2):
            image = load_img(myFile, target_size=(32, 32))
            image = img_to_array(image) 
            data.append(image)
    if source3!=None:
        for myFile in glob.glob (source3):
            image = load_img(myFile, target_size=(32, 32))
            image = img_to_array(image) 
            data.append(image)   
    data=np.array(data)
    return data
def getnpfeatures1(source1,file_names,N): 
    """
    ---input--
    This function is used only on the original raw patches
    source1 paths to subset_training folder
    N- desired number of patches taken from the folder    
    ---returns--
    returns numpy arrays of features
    """
    data=[]
    i=0
    for myFile in file_names:   
        if i!=N:
            image = load_img(source1+myFile, target_size=(32, 32))
            image = img_to_array(image) 
            data.append(image)
            i+=1
    data=np.array(data)
    return data
def getlabels1(files,N): 
    """
    --inputs--
    source1 path to the image folders
    --returns--
    extracts labels from the image name
    """
    labels=[]
    i=0
    for file in files:   
        if i!=N:
            file_name=os.path.splitext(os.path.basename(file))[0]
            label=int(file_name.split('_')[4][5])
            #print(file_name)
            #print(label)
            labels.append(label) 
            i+=1
            #print(label)
    labels=np.array(labels)
    print('labeels',labels)
    return labels
if __name__=="__main__":    
   
    #-------------------------------option 1: train on full data with original labels, specify your path --------------------
    """
    y_test=getlabels('/home/ubuntu/preprocessing/maincode/files/testing/')
    x_test=getnpfeatures('/home/ubuntu/preprocessing/maincode/files/testing/*.png')
    x_val=getnpfeatures('/home/ubuntu/preprocessing/maincode/files/validation/*.png')
    y_val=getlabels('/home/ubuntu/preprocessing/maincode/files/validation/')
    y_train=getlabels('/home/ubuntu/preprocessing/maincode/files/subset_training/','/home/ubuntu/preprocessing/maincode/files/training/')
    x_train=getnpfeatures('/home/ubuntu/preprocessing/maincode/files/subset_training/*.png','/home/ubuntu/preprocessing/maincode/files/training/*.png') 
    model=load_model('original_labels.h5',custom_objects={'auc': auc})
    """
    #----------------------------------------------------------------------------------------------
    
    
    #------------- option 2: this example is if you want to train on the desired number of training samples from the set that is assumed to be label(subset_training)
    n=6000  # train on the desired number of instances
    print('when n is ', n)
    path='/home/ubuntu/preprocessing/maincode/files/subset_training/' #path to your training instances
    file_names= os.listdir(path)  
    p = np.random.RandomState(seed=1).permutation(len(file_names))
    file_names=np.array([file_names[i] for i in p])
    y_train=getlabels1(file_names, n)
    x_train=getnpfeatures1('/home/ubuntu/preprocessing/maincode/files/subset_training/',file_names,n)        
    #x_train,x_test,y_train,y_test= train_test_split(
    #                x_train,y_train, test_size=0.30,random_state=42)
    x_train,x_val,y_train,y_val= train_test_split(
                    x_train,y_train, test_size=0.20,random_state=42)
    y_test=getlabels('/home/ubuntu/preprocessing/maincode/files/testing/')
    x_test=getnpfeatures('/home/ubuntu/preprocessing/maincode/files/testing/*.png')
    with open('original_labels.json', 'r') as json_file:  #the best architecture found by nni
        loaded_model_json = json_file.read()
    model = build_graph_from_json(loaded_model_json)
    optimizer = SGD(lr=0.001, momentum=0.9, decay=1e-5)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=[auc]) 
    #----------------------------------------------------------------------------------------------------------------
    x_test = x_test.astype("float32")
    x_train = x_train.astype("float32")
    y_train = to_categorical(y_train, 2)            
    x_val = x_val.astype("float32")
    y_val = to_categorical(y_val, 2)
    x_train /= 255.0
    x_test /= 255.0
    x_val /=255.0
    avg_auc=0
    avg_acc=0
    pr=0
    rc=0
    f1=0
    #-------------------------- comment it out if you use option 1 ----------------------------------------
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=32,
        validation_data=(x_val, y_val),
        epochs=80, 
        shuffle=True,
        callbacks=[            
            EarlyStopping(min_delta=0.001, patience=10)
        ],
        ) 
    #-------------------------------------------------------------------------------------------------------
    preds_te=model.predict(x_test)
    print(preds_te)
    preds_te=np.argmax(preds_te,axis=1)
    accuracy_test = accuracy_score(y_test,preds_te)
    print('acc',accuracy_test)
    avg_acc=accuracy_test
    auc=roc_auc_score(y_test,preds_te)
    print('auc',auc)
    avg_auc=auc
    f1=f1_score(y_test,preds_te)
    pr=precision_score(y_test,preds_te)
    rc=recall_score(y_test,preds_te)
    conf_mat=confusion_matrix(y_test,preds_te)
    print('final auc',avg_auc)
    print('final acc',avg_acc)
    print('final pr',pr)
    print('final rc',rc)
    print('final f1',f1)
    conf_mat=confusion_matrix(y_test,preds_te)
    print('conf matrix of newly labeled',conf_mat)
    model.save('original_labels_'+str(n)+'.h5')
    model.save_weights('original_labels_w_'+str(n)+'.h5')
    #plot_model(model,show_shapes=True,to_file='alllabels_endmodel.png')
