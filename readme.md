# An Interpretable Semi-Supervised Framework for Patch-Based Classification of Breast Cancer

## Summary

We present a novel semi-supervised learning framework for IDC detection using small amounts of labelled training examples to take advantage of cheap available unlabeled data. To gain trust in the prediction of the proposed framework, we explain the prediction globally. Our proposed framework consists of five main stages: data augmentation, feature selection, dividing co-training data labelling, deep neural network modelling, and the interpretability of neural network prediction. The data cohort used in this study contains digitized BCa histopathology slides from 162 women with IDC at the Hospital of the University of Pennsylvania and the Cancer Institute of New Jersey. To evaluate the effectiveness of the deep neural network model used by the proposed approach, we compare it to different state-of-the-art network architectures; AlexNet and a shallow VGG network trained only on the labelled data. The results show that the deep neural network used in our proposed approach outperforms the state-of-the-art techniques achieving balanced accuracy of 0.73 and F-measure of 0.843. In addition, we compare the performance of the proposed semi-supervised approach to state-of-the-art semi-supervised DCGAN technique and self-learning technique. The experimental evaluation shows that our framework outperforms both semi-supervised techniques and detects IDC with an accuracy of 85.75\%, a balanced accuracy of 0.865, and an F-measure of 0.773 using only 10\% labelled instances from the training dataset while the rest of the training dataset is treated as unlabeled.

## Dataset:
For IDC detection, we use a data cohort that consists of digitized Bca histopathology slides obtained from 162 women diagnosed with IDC at the Cancer Institute of New Jersey and the hospital of the University of Pennsylvania. The cohort was randomly split into three subsets, including 84 for training, 29 for validation and 49 for evaluation. The patch-based dataset used in this study obtained from the original cohort consists of 82,883 patches for training, 31,352 patches for validation (i.e. 114,235 patches for full training) and 50,963 instances for testing(http://www.andrewjanowczyk.com/use-case-6-invasive-ductal-carcinoma-idc-segmentation/). From the 114,235 patches of the entire training dataset, we randomly select 12k instances (6k instances from each class) as training data, and we remove the ground truth of the rest of the entire training patches obtaining 102,235 unlabelled patches. 

The concepts used by the global interpretability technique are extracted from the nuclear segmentation dataset[1]. The nuclear segmentation dataset consists of 30 whole slide images of digitized tissue samples of several organs obtained from 18 different hospitals. The dataset contains nuclear appearances of seven different organs, including breast, liver, kidney, prostate, colon, stomach, and bladder. Since computational requirements for processing WSIs are high, a sub-image of 1000$\times$1000 is cropped from WSIs, and more than 21,000 nuclear boundaries are annotated in \texttt{Aperio ImageScope}. Only five images are extracted that contain the nuclear appearance of breast cancer which are then segmented into 100 patches in which the concepts are extracted.

## How to use 
Each patch has its name: 10253_idx5_x1351_y1101_class0.png, thus labels are assigned itself on the name 
of each patch. After you download it, you need to construct Training/testing/validation sets like it is in the baseline 
paper. For this you need to specify your paths in the code and run the following command in the dataset 
directory:
 
python3 test_train_valisplit



In this study we assume that only 10% of the data is labeled and the rest is unlabeled. We select 10% of available
training and validation instances, that are 12,000 image patches and use them to train our model to label another 90%
of images. Thus, you need to make another folder, we call it subset_training.
We randomly selected 6000 positive and 6000 negative samples from the training folder and put them in the subset_training folder.
This can be also done by calling python3 dataset/balance_classes.py. Here you will need to specify your own origin,destination folders.

After you make this folder, following codes were used to try different labeling methods:

10gb_labelling_vgg_gan.py 

Data Augmentation

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

### Modelling
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
### Interpretability

Model interpretability: code is based on this work https://github.com/medgift/iMIMIC-RCVs.

## Other dependencies:
All the required packages should be already installed if you install nni requirements.txt in nni repo.
If you need to run interpretability, you will need you install packages mentioned in iMIMIC-RCVs/requirements.txt

## Refrences
[1] Kumar, Neeraj, Ruchika Verma, Sanuj Sharma, Surabhi Bhargava, Abhishek Vahadane, and Amit Sethi. "A dataset and a technique for generalized nuclear segmentation for computational pathology." IEEE transactions on medical imaging 36, no. 7 (2017): 1550-1560

    
    
