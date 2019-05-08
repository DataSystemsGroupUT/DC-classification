This directory is used to train the best neural architecture found by NNI.

There are two cases:
a) You train on the originally labeled data
b) You train on the data that is labeled by us

in case of a)
original_labels_w.h5 : here are saved the neural network weights
original_labels.h5  : here are saved both architecture and weights
original_label.json : this is a neural architecture only in case you want to fit on your data

you can run the script by running python3 original_labels.py
but you will have to change your own path to the training and testing folders and the number of training samples you want to use
for training. Follow the instructions in the code for other details.

In case of b):

.h5 have the same meaning as it had in case of a.
you can run the script by running python3 newly_labeled.py
you will have to only specify your input path


