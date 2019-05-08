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
from keras.optimizers import Adagrad
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
random.seed(1)
def build_graph_from_json(ir_model_json):
    """build model from json representation
    """
    graph = json_to_graph(ir_model_json)
    model = graph.produce_keras_model()
    return model
def getlabels(source1,source2=None,): #each image_name contains the label as a last symbol. This function takes the destination
    labels=[]#of the images and returns the labels of that image set
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
def getlabels2(source1,source2=None,): #each image_name contains the label as a last symbol. This function takes the destination
    labels=[]#of the images and returns the labels of that image set
    files= os.listdir(source1)  
    files2=os.listdir(source2)  
    i=0
    for file in files:   
        if i!=3000:
            file_name=os.path.splitext(os.path.basename(file))[0]
            label=int(file_name.split('_')[4][5])
            labels.append(label)
            i+=1            
    labels=np.array(labels)
    return labels
def getnpfeatures2(source1,source2=None,vgg_names1=None, vgg_names2=None): #takes in the destination of the images, reads them and saves as the corresponding np.arrays
    data=[]
    files= os.listdir(vgg_names1)  
    files2=os.listdir(vgg_names2)
    i=0
    for file in files:   #training data input dimension is resized from
        if i!=3000:
            myFile=os.path.splitext(os.path.basename(file))[0]
            image = load_img(source1+myFile+'.png', target_size=(32, 32)) 
            image = img_to_array(image) 
            data.append(image)
            i+=1
    if source2!=None:
        for file in files2:
            myFile=os.path.splitext(os.path.basename(file))[0]
            image = load_img(source2+myFile+'.png', target_size=(32, 32))
            image = img_to_array(image) 
            data.append(image)
    data=np.array(data)
    return data
def getnpfeatures(source1,source2=None,vgg_names1=None, vgg_names2=None): #takes in the destination of the images, reads them and saves as the corresponding np.arrays
    data=[]
    files= os.listdir(vgg_names1)  
    files2=os.listdir(vgg_names2)
    for file in files:   #training data input dimension is resized from
        myFile=os.path.splitext(os.path.basename(file))[0]
        image = load_img(source1+myFile+'.png', target_size=(32, 32)) 
        image = img_to_array(image) 
        data.append(image)
    if source2!=None:
        for file in files2:
            myFile=os.path.splitext(os.path.basename(file))[0]
            image = load_img(source2+myFile+'.png', target_size=(32, 32))
            image = img_to_array(image) 
            data.append(image)
    data=np.array(data)
    return data
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

if __name__=="__main__": 
    #---------------------- load your model --------------------------------------------------------
    model=load_model('newly_labeled.h5',custom_objects={'auc': auc}) #load the model trained on the newly labeled data
    #----------------------- specify your input data -----------------------------------------------
    test_y=getlabels('/home/ubuntu/preprocessing/maincode/files/testing_features/VGG-1-224/')  #the way it's sorted in vgg file
    test_x=getnpfeatures('/home/ubuntu/preprocessing/maincode/files/testing/',None,'/home/ubuntu/preprocessing/maincode/files/testing_features/VGG-1-224/')
    val_x=getnpfeatures('/home/ubuntu/preprocessing/maincode/files/validation/',None,'/home/ubuntu/preprocessing/maincode/files/validation_features/VGG-1-224/')
    train_y_part1=getlabels2('/home/ubuntu/preprocessing/maincode/files/subset_features/VGG-1-224/')
    train_x=getnpfeatures2('/home/ubuntu/preprocessing/maincode/files/subset_training/','/home/ubuntu/preprocessing/maincode/files/training/','/home/ubuntu/preprocessing/maincode/files/subset_features/VGG-1-224/','/home/ubuntu/preprocessing/maincode/files/training_features/VGG-1-224/')   
    f = open("/home/ubuntu/preprocessing/maincode/files/masterthesis/labels/vgg_gan", 'rb')  
    labels=pickle.load(f)
    leng=labels.shape[0]-val_x.shape[0]  #how many instances from training folder were predicted? note that labels=num_training+num_validation
    train_y_part2=labels[:leng]
    val_y=labels[leng:]
    train_y=np.append(train_y_part1,train_y_part2)
    #-------------------------------------------------------------------------------------------------
    x_test = test_x.astype("float32")
    x_train = train_x.astype("float32")
    y_train = to_categorical(train_y, 2)            
    x_val = val_x.astype("float32")
    y_val = to_categorical(val_y, 2)
    x_train /= 255.0
    x_test /= 255.0
    x_val/=255.0
    print('val shape', val_x.shape, val_y.shape)
    print('train_x shape', train_x.shape, train_y.shape)
    print('test_x shape', test_x.shape, test_y.shape)
    optimizer = SGD(lr=0.001, momentum=0.9, decay=1e-5)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=[auc])  
    avg_auc=0
    avg_acc=0
    pr=0
    rc=0
    f1=0
    """ 
    #----------------------- this is for fitting the model-------------------------------
   
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=32,  #32,
        validation_data=(x_val, y_val),
        epochs=44,  
        shuffle=True,
        #callbacks=[            
        #    EarlyStopping(min_delta=0.001, patience=10)
        #],
        )
    #-----------------------------------------------------------------------------------
    """
    preds_te=model.predict(x_test)
    preds_te=np.argmax(preds_te,axis=1)
    accuracy_test = accuracy_score(test_y,preds_te)
    print('acc',accuracy_test)
    avg_acc=accuracy_test
    auc=roc_auc_score(test_y,preds_te)
    print('auc',auc)
    avg_auc=auc
    f1=f1_score(test_y,preds_te)
    pr=precision_score(test_y,preds_te)
    rc=recall_score(test_y,preds_te)
    conf_mat=confusion_matrix(test_y,preds_te)
    print('final auc',avg_auc)
    print('final acc',avg_acc)
    print('final pr',pr)
    print('final rc',rc)
    print('final f1',f1)
    conf_mat=confusion_matrix(test_y,preds_te)
    print('conf matrix of newly labeled performance',conf_mat)
    #model.save('mix_data_'+str(3000)+'.h5')
    #model.save_weights('mix_data_w_'+str(3000)+'.h5')
    #plot_model(model,show_shapes=True,to_file='newlylabels_endmodel.png')
