"""
!!! P.S following code is a modified version of an online source code that I changed but could not find the source to cite it any more. comment it if you find it.

This python file trains DCGAN and generates required number of synthetic images.
you have to train it separately per each class. The following example generates class 1 examples from a binary image dataset.
model takes the images of shape 32x32x3
If you want to train on your own dataset change the following blocks:
in input block : define your own feature set and labels in np.arrays
in the details block : change number of images you want to generate and specify the class on what you want to trin DCGAN,
besides, provide your own paths for epoch results and final results.
in the specify the class block: add small changes like some if/else constructions if you have more than 2 classes, or other classes than 1/0.

And finally, enjoy the magic of GAN :)))) 

P.s Training for 1200 epochs for both classes took around ten days on a machine without GPU. You can decrease the number of epochs.

"""
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LeakyReLU, BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose, Reshape, Flatten
from keras.optimizers import Adam
from keras import initializers
from keras.utils import plot_model, np_utils
from keras import backend as K
import os
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import glob
def getlabels(source1,source2=None): #each image_name contains the label as a last symbol. This function takes the destination
    """
    --inputs--
    source1,source2- path to the image folders
    --returns--
    extracts labels from the image name
    """
    print('labels called')
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

def getnpfeatures(source1,source2=None,source3=None):
    """
    --inputs--
    source1,source2- path to the image folders
    --returns--
    returns np.array of images from each folder
    """
    print('npfeaturescalled')
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
if __name__=="__main__": 
    #------------------------   input block #----------------------------------------
    # you don't need those function calls if you work on another data. Just plug in your feature set and label set.
    y=getlabels('/home/ubuntu/preprocessing/maincode/files/subset_training/')
    num_classes = len(np.unique(y))
    x=getnpfeatures('/home/ubuntu/preprocessing/maincode/files/subset_training/*.png') 
    print('feature_calling finished')
    #----------------------------------- details block -------------------------------
    samples=2000# numer of syntetic images generated in the end
    train_n=6000 #number of images to be trained on GAN  (that number of images will be taken from your training set for particular class)
    class_=1 #which class are you going to train? 
    path_to_epochs='1_epochs/' #path to epochs file- synthetic images generated after every 50 epochs.
    path_to_synth='1_fakes/' #were do you wanna generate final synthetic images?
    epochs = 1200   # you can reduce.
    batch_size = 16
    smooth = 0.1
    #-----------------------------------specify class-------------------------------------
    """
    this block takes specified number(not more than number of samples of that class in your training data)
    of images from your training set for specified class. 
    if you have a binary dataset with int labels like 1 and 0 leave it, if not then you need to add some if/else construction for extra classes
  
    """
    X_train=[]
    it=0
    if class_==1:    #if you choose to train on class 1 generate specified number of class 1
        print('class is one')
        y1=np.ones(train_n)  
        print(len(y1))
    else:
        y1=np.zeros(train_n)   #or if it was a class zero, make zero labels   # add extra classes information here with more else constructions 
    for i in range(0,len(y)):  # in case you have multi classes.
        print(i)
        if(it!=train_n):
            if y[i]==class_: 
                X_train.append(x[i]) # we also need features from your training set :)))
                it+=1
    X_train=np.array(X_train)  
    #-----------------------------------------prints dataset shapes and splits into train/test-------------------------------------------------------    
    print('train dims before splitting ',X_train.shape,'  ', y1.shape )
    X_train, X_test, y_train, y_test = train_test_split(
                    X_train, y1, test_size=0.2,random_state=42)        
    print('test dims',X_test.shape,'  ', y_test.shape )
    print('train dims after splitting ',X_train.shape,'  ', y_train.shape )
    #------------ reshaping and normalizing the input-----------------#
    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 3, 32, 32)
        X_test = X_test.reshape(X_test.shape[0], 3, 32, 32)
        input_shape = (3, 32, 32)
    else:
        X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
        X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
        input_shape = (32, 32, 3)    
    # convert class vectors to binary class matrices
    # the generator is using tanh activation, for which we need to preprocess 
    # the image data into the range between -1 and 1.

    X_train = np.float32(X_train)
    X_train = (X_train / 255 - 0.5) * 2
    X_train = np.clip(X_train, -1, 1)

    X_test = np.float32(X_test)
    X_test = (X_train / 255 - 0.5) * 2
    X_test = np.clip(X_test, -1, 1)

    print('X_train reshape:', X_train.shape)
    print('X_test reshape:', X_test.shape)
    
    #----------------------------#Generator#---------------------------------#
    latent_dim = 100

    init = initializers.RandomNormal(stddev=0.02)

    # Generator network
    generator = Sequential()

    # FC: 2x2x512
    generator.add(Dense(2*2*512, input_shape=(latent_dim,), kernel_initializer=init))
    generator.add(Reshape((2, 2, 512)))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))

    # # Conv 1: 4x4x256
    generator.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding='same'))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))

    # Conv 2: 8x8x128
    generator.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same'))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))

    # Conv 3: 16x16x64
    generator.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding='same'))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))

    # Conv 4: 32x32x3
    generator.add(Conv2DTranspose(3, kernel_size=5, strides=2, padding='same',
                                  activation='tanh'))
    # prints a summary representation of your model
    generator.summary()            
    #-----------------------#Discriminator#------------------------------------#
    # prints a summary representation of your model
    
    # imagem shape 32x32x3
    img_shape = X_train[0].shape
    # Discriminator network
    discriminator = Sequential()
    # Conv 1: 16x16x64
    discriminator.add(Conv2D(64, kernel_size=5, strides=2, padding='same',
                             input_shape=(img_shape), kernel_initializer=init))
    discriminator.add(LeakyReLU(0.2))

    # Conv 2:
    discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))

    # Conv 3: 
    discriminator.add(Conv2D(256, kernel_size=5, strides=2, padding='same'))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))

    # Conv 3: 
    discriminator.add(Conv2D(512, kernel_size=5, strides=2, padding='same'))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))

    # FC
    discriminator.add(Flatten())

    # Output
    discriminator.add(Dense(1, activation='sigmoid'))


    discriminator.summary()
    #---- compile the model-----------#
    # Optimizer
    discriminator.compile(Adam(lr=0.0003, beta_1=0.5), loss='binary_crossentropy',
                      metrics=['binary_accuracy'])
    #-----Combined network------------#    
    #We connect the generator and the discriminator to make a DCGAN.
    discriminator.trainable = False
    z= Input(shape=(latent_dim,))
    img = generator(z)
    decision = discriminator(img)
    d_g = Model(inputs=z, outputs=decision)
    d_g.compile(Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy',
            metrics=['binary_accuracy'])
    # prints a summary representation of your model
    d_g.summary()
    #------Fit model-----------------#

    real = np.ones(shape=(batch_size, 1))
    fake = np.zeros(shape=(batch_size, 1))

    d_loss = []
    g_loss = []

    for e in range(epochs + 1):
        for i in range(len(X_train) // batch_size):         
            # Train Discriminator weights
            discriminator.trainable = True
            
            # Real samples
            X_batch = X_train[i*batch_size:(i+1)*batch_size]
            d_loss_real = discriminator.train_on_batch(x=X_batch,
                                                       y=real * (1 - smooth))
            
            # Fake Samples
            z = np.random.normal(loc=0, scale=1, size=(batch_size, latent_dim))
            X_fake = generator.predict_on_batch(z)
            d_loss_fake = discriminator.train_on_batch(x=X_fake, y=fake)
             
            # Discriminator loss
            d_loss_batch = 0.5 * (d_loss_real[0] + d_loss_fake[0])
            
            # Train Generator weights
            discriminator.trainable = False
            g_loss_batch = d_g.train_on_batch(x=z, y=real)

            print(
                'epoch = %d/%d, batch = %d/%d, d_loss=%.3f, g_loss=%.3f' % (e + 1, epochs, i, len(X_train) // batch_size, d_loss_batch, g_loss_batch[0]),
                100*' ',
                end='\r'
            )
        
        d_loss.append(d_loss_batch)
        g_loss.append(g_loss_batch[0])
        print('epoch = %d/%d, d_loss=%.3f, g_loss=%.3f' % (e + 1, epochs, d_loss[-1], g_loss[-1]), 100*' ')
        if e % 100==0:  #after every 50 epochs, this will generate sample images and you can monitor the progress over epochs in the end
            e_samples = 10
            x_fake = generator.predict(np.random.normal(loc=0, scale=1, size=(samples, latent_dim)))
            for k in range(e_samples):
                plt.subplot(2, 5, k + 1, xticks=[], yticks=[])
                plt.imshow(((x_fake[k] + 1)* 127).astype(np.uint8))
            plt.tight_layout()
            plt.savefig(path_to_epochs+'fakes_for_epoch'+str(e)+'.png', bbox_inches="tight")
    x_fake = generator.predict(np.random.normal(loc=0, scale=1, size=(samples, latent_dim)))
    for k in range(samples):
        plt.imsave(path_to_synth+str(k)+'.png',((x_fake[k] + 1)* 127).astype(np.uint8))

        
