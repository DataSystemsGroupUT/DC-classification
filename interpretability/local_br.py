"""
This script calculates local br scores of
3 positive and 3 negative examples. for each image,
it calculates 100 closest neighbors and based on them,
average sensitivity and br scores are measured. The concepts are assumed
to be calculated to run this script.
"""
import sys
sys.path.append('./scripts/keras_vis_rcv')
sys.path.append('./scripts/models')
import rcv_utils
import os
import numpy as np
import keras.backend as K
from PIL import Image
import tensorflow as tf
from importlib import reload
import matplotlib.pyplot as plt
import scipy.stats
import h5py 
from scipy import spatial
import pandas as pd 
from sklearn.decomposition import PCA
import warnings
from keras.utils import multi_gpu_model, to_categorical
from keras.models import model_from_json
from nni.networkmorphism_tuner.graph import json_to_graph
from keras.models import load_model
import glob
import os
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc
def getnpfeatures(source1,source2=None,source3=None):
    """
    used when applying on the original raw patches
    --inputs--
    source1,source2- path to the image folders
    --returns--
    returns np.array of images from each folder
    """
    print('features called')
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
def build_graph_from_json(ir_model_json):
    """build model from json representation
    """
    graph = json_to_graph(ir_model_json)
    model = graph.produce_keras_model()
    return model
if __name__=="__main__": 
    
    warnings.filterwarnings('ignore') 
    bc_model=load_model('/home/ubuntu/preprocessing/maincode/files/masterthesis/nni/newly_labeled.h5',custom_objects={'auc': auc})    
    concepts=[
     'contrast', 
     'ASM',
     'correlation',
     ]   
    max_rep = 1     
    #--------------------- Sensitivity scores -----------------------------------------------
    test_x=getnpfeatures('/home/ubuntu/preprocessing/maincode/files/testing/*.png')
    imgs=getnpfeatures('/home/ubuntu/preprocessing/maincode/files/DC-classification/interpretability/images_1/*.png')
    imgs/=255.0
    imgs=np.reshape(imgs,(-1,3072))
    pca_test_x=test_x/255.0
    pca_test_x=np.reshape(pca_test_x,(-1,3072))
    pca = PCA(.90)
    pca.fit(pca_test_x)
    pca_test_x= pca.transform(pca_test_x)
    imgs=pca.transform(imgs)
    print('pcaperformed')
    tree = spatial.KDTree(pca_test_x)
    for i in imgs[:1]:
        tr=np.array(tree.query(i,k=101))
        dists=np.array(tr[0])
        dists=np.delete(dists,0)
        dists=np.multiply(dists,-1)
        dists=np.exp(dists)
        ids=np.array(tr[1])#number of neighbors
        ids=np.delete(ids,0)# delete the first id, as it's the length to itsels the same image           
        ids=ids.astype(int)
        test_x=test_x[ids,:]
        normalised_tumor_patches = np.array([np.uint8(patch) for patch in test_x])
        test_inputs = np.float32(normalised_tumor_patches)
        test_inputs/=255.0
        for concept in concepts:
            print(concept)
            for i in range(0, max_rep):
                rcv = np.load('./rcv/rcv_'+concept+'_'+str(i)+'.npy',allow_pickle=True)
                rcv /= np.linalg.norm(rcv)
                repetition = 0
                scores=[]
                for p in range(len(test_inputs)):
                    nnn = rcv_utils.compute_tcav(bc_model,-1,0, np.expand_dims(test_inputs[p], axis=0), wrt_tensor=bc_model.layers[12].output)  #is unda iyos rac maglaa
                    flattened_derivative=nnn.ravel()
                    score = np.dot(flattened_derivative, rcv) #sensitivity scores
                    #
                    #score=np.dot(flattened_derivative, rcv)
                    scores.append(score)
                    filet=open('rcv_'+concept+'_'+str(i)+'.txt', 'a')
                    filet.write(str(repetition)+','+str(score)+'\n')
                    filet.close()  
                score=np.dot(dists,scores)                    
                print('avg sensitivity for concept: ', concept,sum(scores)/len(scores))
        reload(rcv_utils)
        mmax=1
        brs=[]
        avg_brs=[]
        for concept in concepts:
           #for each concept, take me average values         
            for i in range(0,mmax):
                if os.path.exists('./'+'rcv_'+concept+'_'+str(i)+'.txt'):
                    df=[]
                    df=pd.read_csv('./'+'rcv_'+concept+'_'+str(i)+'.txt', header=None)
                    R = np.load('./rcv/reg_score_'+concept+'_'+str(i)+'.npy',allow_pickle=True)
                    br = R * np.mean(np.array(df[1])) / np.std(np.array(df[1]))
                    #print(concept,' ', br)
                    brs.append(br)
        print(brs)
        brs=2*(brs-min(brs))/(max(brs)-min(brs))-1
        print('brs',brs)
