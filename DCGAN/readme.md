Following directory is performing Keras DCGAN to generate new synthetic samples. You need to run it separately for each class. In this case
for class 1 and class 0. Input images have to be the dimension of 32x32x3 and in 2 separate np.arrays(one for features, another for labels)
To run on your own dataset or change other details(on how many number of images from your training set you wanna train it, DCGAN.py 
file provides detailed explanations about how to modify it.

to run it: simply run python3 DCGAN.py on your command window