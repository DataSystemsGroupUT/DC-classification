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
import pandas as pd 
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
        #print(myFile)    
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
    #---------------------------------- calculate correlations between model and patches--------------------
    patches, masks, nuclei, stats = rcv_utils.get_norm_patches(path='/home/ubuntu/preprocessing/maincode/files/masterthesis/iMIMIC-RCVs/data/datasets/training/0/')
    patches = [np.uint8(x) for x in patches]
    patches=np.array(patches)
    nuclei_morph = rcv_utils.nuclei_morphology(stats)
    nuclei_text = rcv_utils.nuclei_texture(patches, nuclei) 
    patches=np.float32(patches)
    patches/=255.0
    predictions = bc_model.predict(patches)
    predictions=predictions[:,0]
    predictions = predictions.reshape((len(predictions),))
    print(predictions.shape)
    np.save('results/predictions', predictions)
    def corr_analysis(feature, pred):
        return scipy.stats.pearsonr(np.array(feature), np.array(pred))
    # Correlation analysis of texture features
    print ('contrast: ', corr_analysis(nuclei_text['contrast'], predictions))
    print ('ASM: ', corr_analysis(nuclei_text['ASM'], predictions))
    print ('correlation: ', corr_analysis(nuclei_text['correlation'], predictions))    
    #--------------------------------------------------------------------------------------------------------
    concepts=[
     'contrast', 
     'ASM',
     'correlation',
     ]    
    
    for i in range(1,len(bc_model.layers)):   
        print(i)
        max_rep = 1
        generate_patches = False
        if not generate_patches:
            max_rep=1
        for repetition in range(0, max_rep):
            patches, masks, nuclei, stats = rcv_utils.get_norm_patches(path='./data/datasets/training/0/')
            if generate_patches:
                tr_set=rcv_utils.get_cv_training_set('/home/ubuntu/preprocessing/maincode/files/masterthesis/iMIMIC-RCVs/data/datasets/breast_nuclei/Tissue Images/', repetition) 
                patches, masks, nuclei, stats = rcv_utils.get_norm_patches(path='/home/ubuntu/preprocessing/maincode/files/masterthesis/iMIMIC-RCVs/data/datasets/training/'+str(repetition)+'/')
            patches = [np.uint8(x) for x in patches]
            patches=np.array(patches)
            #nuclei_morph = rcv_utils.nuclei_morphology(stats)
            # Nuclei texture statistics
            nuclei_text = rcv_utils.nuclei_texture(patches, nuclei)
            input=np.float32(patches)
            input /= 255.0 
            get_layer_output = K.function([bc_model.layers[0].input],
                                      [bc_model.layers[i].output])   #get_layer tu saxeli ici
            feats = get_layer_output([input])
            if not os.path.exists('./rcv/'):
                os.mkdir('./rcv/')
                if not os.path.exists('./rcv/phis/'):
                    os.mkdir('./rcv/phis/')
            np.save('./rcv/phis/'+str(repetition)+'_concepts_phis_'+str(1), np.asarray(feats[0]))
        feats=[]
        for repetition in range(max_rep):
            patches, masks, nuclei, stats = rcv_utils.get_norm_patches(path='./data/datasets/training/0/')#str(repetition)+'/')
            patches = [np.uint8(x) for x in patches]
            patches=np.array(patches)
            nuclei_text = rcv_utils.nuclei_texture(patches, nuclei)
            feats=np.load('./rcv/phis/'+str(repetition)+'_concepts_phis_1.npy')  #nuclear patch features extracted from layer l
            X=(np.asarray([x.ravel() for x in feats], dtype=np.float64))
            for c in concepts[-3:]:
                reg_score, cv = rcv_utils.solve_regression(X, np.asarray(nuclei_text[c]))  #score
                np.save('./rcv/reg_score_'+c+'_'+str(repetition)+'.npy', reg_score)  #direction to increasing values
                np.save('./rcv/rcv_'+c+'_'+str(repetition)+'.npy', cv)              
                print('layer',' ',i,' reg score', ' ', c, reg_score)
    """        
    #--------------------- Sensitivity scores -----------------------------------------------
    test_x=getnpfeatures('/home/ubuntu/preprocessing/maincode/files/testing/*.png')
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
            for p in range(len(test_inputs[:100])):
                nnn = rcv_utils.compute_tcav(bc_model,-1,0, np.expand_dims(test_inputs[p], axis=0), wrt_tensor=bc_model.layers[12].output)  #is unda iyos rac maglaa
                flattened_derivative=nnn.ravel()
                #print('derivative shape',flattened_derivative.shape)
                #print('rcv shape', rcv.shape)
                score = np.dot(flattened_derivative, rcv) #sensitivity scores
                #score=np.dot(flattened_derivative, rcv)
                scores.append(score)
                #print('sensityvity score.shape',score.shape, 'scores',score)
                filet=open('rcv_'+concept+'_'+str(i)+'.txt', 'a')
                filet.write(str(repetition)+','+str(score)+'\n')
                filet.close()   
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
                print(concept,' ', br)
                brs.append(br)
    print(brs)
    brs=2*(brs-min(brs))/(max(brs)-min(brs))-1
    print('brs',brs)
    """