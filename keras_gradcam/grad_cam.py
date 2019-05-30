import sys
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import tensorflow as tf
from tensorflow.python.framework import ops
from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from keras.models import load_model
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import pickle
import cv2
from keras.models import model_from_json
from keras.utils import multi_gpu_model, to_categorical
import os
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.metrics import confusion_matrix
from nni.networkmorphism_tuner.graph import json_to_graph
from keras.callbacks import EarlyStopping, TensorBoard
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
from sklearn.metrics import roc_auc_score
# Define model here ---------------------------------------------------
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def build_model():
    """Function returning keras model instance.
    
    Model can be
     - Trained here
     - Loaded with load_model
     - Loaded from keras.applications
    """
    #return VGG16(include_top=True, weights='imagenet')
    model=load_model('/home/ubuntu/preprocessing/maincode/files/masterthesis/nni/newly_labeled.h5',custom_objects={'auc': auc})
    return model
H, W = 32, 32 # Input shape, defined by the model (model.input_shape)
# ---------------------------------------------------------------------

def load_image(path, preprocess=True):
    """Load and preprocess image."""
    x = image.load_img(path, target_size=(H, W))
    if preprocess:
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
    return x


def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    x = x.copy()
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    """Utility function to normalize a tensor by its L2 norm"""
    return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)


def build_guided_model():
    """Function returning modified model.
    
    Changes gradient function for all ReLu activations
    according to Guided Backpropagation.
    """
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                   tf.cast(op.inputs[0] > 0., dtype)

    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        new_model = build_model()
    return new_model


def guided_backprop(input_model, images, layer_name):
    """Guided Backpropagation method for visualizing input saliency."""
    input_imgs = input_model.input
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(layer_output, input_imgs)[0]
    backprop_fn = K.function([input_imgs, K.learning_phase()], [grads])
    grads_val = backprop_fn([images, 0])[0]
    return grads_val


def grad_cam(input_model, image, cls, layer_name):
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output[0, cls]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]
    # Normalize if necessary
    # grads = normalize(grads)
    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # Process CAM
    cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam

def grad_cam_batch(input_model, images, classes, layer_name):
    """GradCAM method for visualizing input saliency.
    Same as grad_cam but processes multiple images in one run."""
    loss = tf.gather_nd(input_model.output, np.dstack([range(images.shape[0]), classes])[0])
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(loss, layer_output)[0]
    gradient_fn = K.function([input_model.input, K.learning_phase()], [layer_output, grads])

    conv_output, grads_val = gradient_fn([images, 0])    
    weights = np.mean(grads_val, axis=(1, 2))
    cams = np.einsum('ijkl,il->ijk', conv_output, weights)
    
    # Process CAMs
    new_cams = np.empty((images.shape[0], W, H))
    for i in range(new_cams.shape[0]):
        cam_i = cams[i] - cams[i].mean()
        cam_i = (cam_i + 1e-10) / (np.linalg.norm(cam_i, 2) + 1e-10)
        new_cams[i] = cv2.resize(cam_i, (H, W), cv2.INTER_LINEAR)
        new_cams[i] = np.maximum(new_cams[i], 0)
        new_cams[i] = new_cams[i] / new_cams[i].max()
    
    return new_cams


def compute_saliency(model, guided_model, input, filename,layer_name='block5_conv3', cls=-1, visualize=True, save=True):
    """Compute saliency using all three approaches.
        -layer_name: layer to compute gradients;
        -cls: class number to localize (-1 for most probable class).
    """
    input = np.expand_dims(input, axis=0)
    print('inputshape', input.shape)
    preprocessed_input =input #load_image(img_path)
    preprocessed_input=  preprocessed_input.astype("float32")
    preprocessed_input/=255.0
    predictions = model.predict(preprocessed_input)
    print('predictions',predictions)
    top_n = 2
    #top = decode_predictions(predictions, top=top_n)[0]
    #classes = np.argsort(predic#tions[0])[-top_n:][::-1]
    #print('Model prediction:')
    #for c, p in zip(classes, top):
    #    print('\t{:15s}\t({})\twith probability {:.3f}'.format(p[1], c, p[2]))
    if cls == -1:
        cls = np.argmax(predictions)
    #class_name = decode_predictions(np.eye(1, 2,cls))[0][0][1]
    #print("Explanation for '{}'".format(class_name))
    
    gradcam = grad_cam(model, preprocessed_input, cls, layer_name)
    gb = guided_backprop(guided_model, preprocessed_input, layer_name)
    guided_gradcam = gb * gradcam[..., np.newaxis]

    if save:
        jetcam = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
        jetcam = (np.float32(jetcam) + input) / 2
        #cv2.imwrite('results/gradcam'+filename+'.jpg', np.uint8(jetcam))
        #cv2.imwrite('results/'+filename+str(predictions[0])+'.jpg', deprocess_image(gb[0]))
        cv2.imwrite('imgs_0_results/'+filename+str(predictions[0])+'.jpg', deprocess_image(guided_gradcam[0]))
        #cv2.imwrite('results/'+filename+'.jpg',input)

        
    if visualize:
        plt.figure(figsize=(15, 10))
        plt.subplot(131)
        plt.title('GradCAM')
        plt.axis('off')
        plt.imshow(input[0])
        plt.imshow(gradcam, cmap='jet', alpha=0.5)

       # plt.subplot(132)
        plt.title('Guided Backprop')
        plt.axis('off')
        plt.imshow(np.flip(deprocess_image(gb[0]), -1))
        
        plt.subplot(133)
        plt.title('Guided GradCAM')
        plt.axis('off')
        plt.imshow(np.flip(deprocess_image(guided_gradcam[0]), -1))
        plt.show()
        
    return gradcam, gb, guided_gradcam
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
def getnpfeatures(source1): #takes in the destination of the images, reads them and saves as the corresponding np.arrays
    data=[]
    file_names=[]
    files= os.listdir(source1)  
    for file in files:   #training data input dimension is resized from
        myFile=os.path.splitext(os.path.basename(file))[0]
        file_names.append(myFile)
        image = load_img(source1+myFile+'.png', target_size=(32, 32)) 
        image = img_to_array(image)
        #cv2.imwrite( "origimages/", image);
        data.append(image)
    data=np.array(data)
    file_names=np.array(file_names)
    print(file_names)
    return data,file_names
if __name__ == '__main__':
    model = build_model()
    print(model)
    guided_model = build_guided_model()
    test_y=getlabels('imgs_0/')  #the way it's sorted in vgg file
    test_x,filename=getnpfeatures('imgs_0/')
    for index in range(len(test_x)):
        gradcam, gb, guided_gradcam = compute_saliency(model, guided_model, test_x[index],filename[index],layer_name='conv2d_3',
                                              cls=-1, visualize=True, save=True)
