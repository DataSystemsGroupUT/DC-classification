Invasive Ductal Carcinoma(IDC) is the most aggresive breast cancer. In this github repo, we classify IDC/not IDC 
image patches using semi-supervised learning. 

Dataset:
Original dataset consists 162 whole slide images(WSI). From that 277.524 patches are extracted of size 50x50(198.738
IDC negative and 78.786 IDC positive)
Each patch has its name: 10253_idx5_x1351_y1101_class0.png, thus labels are assigned itself on the name 
of each patch. You can download and find more about the dataset from here:http://www.andrewjanowczyk.com/use-case-6-invasive-ductal-carcinoma-idc-segmentation/
After you download it, you need to construct Training/testing/validation sets like it is in the baseline 
paper. For this you need to specify your paths in the code and run the following command in the dataset 
directory:
 
python3 test_train_valisplit

In this study we assume that only 10% of the data is labeled and the rest is unlabeled. We select 10% of available
training and validation instances, that are 12 000 image patches and use them to train our model to label another 90%
of images. Thus, you need to make another folder, we call it subset_training.
We have selected 6000 positive and 6000 negative samples from the training folder and put them in the subset_training folder.
This can be also done by calling python3 dataset/balance_classes.py. Here you will need to specify your own origin,destination folders.

After you make this folder, following codes were used to try different labeling methods:

10gb_labelling_vgg_gan.py 
    This is the best labeling method that trains DCGAN on the labeled instances and generates synthetic
    images that are added in the training set. After new instances are added in the labeled set, VGG-16
    is adopted to extract the features from all the labeled and unlabeled instances, thus features are 
    extracted from subset_training, the rest of training instances and full validation instances. For 
    synthetic image generation refer to the directory DCGAN. For feature extraction, refer to the VGG_feature_extraction
    directory.
Other labeling methods we have tried:

1gb_labelling.py
    this method traines 1GB on the subset_training instances on the VGG-16 features.
10gb_labelling_vgg.py   
    this method trains 10GB on the subset_training instances, features extracted 
    from  the VGG-16 network.
dif_models_labelling.py
    This file consists of many machine learning classification algorithms. Featres are
    again extracted from VGG-16 pretrained network.
10gb_labelling_pca
    this file traines 10gb models on the dimensionality reduced features using PCA.

After you have labeled all the instances( full training, validation folders):

We perform neural architecture search on the newly labeled data as well as originally labeled
data. Note that subset_training is a subset of the training folder. While training NNI,
we append this subset back to the training folder but the rest of the instances there are
labeled by the above mention algorithm. The full validation folder is also labeled by us.

For neural architecture search, (in case you want to find new architecture for your data)
refer to the NNI github repo:
    https://github.com/Microsoft/nni
otherwise, you still need to install nni using the terminal codes, explained at nni github repo.
After you install nni, you can refer to our nni folder. The model weights are saved to reproduce 
the results. How, is explained in nni/readme.md

Model interpretability: code is based on this work https://github.com/medgift/iMIMIC-RCVs.

Other dependencies:
All the required packages should be already installed if you install nni requirements.txt in nni repo.
If you need to run interpretability, you will need you install packages mentioned in iMIMIC-RCVs/requirements.txt

    

    
    
