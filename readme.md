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
training and validation instances, denoted D_{train} comprises of 12,000 patches and use them to train our model to label another 90%
of images. Thus, you need to make another folder, we call it subset_training.
We randomly selected 6000 positive and 6000 negative samples from the training folder and put them in the subset_training folder.
This can be also done by calling python3 dataset/balance_classes.py. Here you will need to specify your own origin,destination folders.

After you make this folder, following codes were used to try different labeling methods:

10gb_labelling_vgg_gan.py 

### Data Augmentation 

In this work, we employ DCGAN to generate synthetic patches. The DCGAN generator consists of a fully connected layer projecting an input of 100-dimensional uniform distribution to four convolution layers with filter sizes of 256, 128, 64 and 32 and kernel size of 5$\times$5. Except for the output layer, the used activation function is rectified linear unit (Relu). Batch normalization is performed on all layers except for the last one. We train a DCGAN on $D_{train}$ as a preprocessing step. The DCGAN is trained separately on each label using multi-channel image patches containing both the acquired image and the ground truth label. The number of synthetic examples generated for each class is 6k patches. Such 12k generated synthetic patches, denoted $D_{GAN}$, are then used to augment $D_{train}$. DCGAN is trained using Stochastic Gradient Descent as an optimizer for 1200 epochs with the \emph{batch size} of 16 and \emph{learning rate} equal to 0.0003. The loss function applied is a binary cross-entropy. Parameters are the same for both discriminator and generator. For $D_{GAN}$ synthetic patches generation, please refer to the directory DCGAN. 

### Feature extraction

In our experiments, we have considered this approach. Inspired by~\cite{rakhlin2018deep}, in this work, we use a standard pre-trained VGG-16 network for feature extraction. We follow the same procedure for extracting features from both labelled and unlabelled data. We remove the fully connected layers from the VGG network and apply the Global Average Pooling operation to the four internal convolutions layers with 128, 256, 512, and 512 channels, respectively. Next, we concatenate them to form one vector of length 1408. For feature extraction, refer to the VGG_feature_extraction directory. 

### Data Labeling
First, our procedure starts by shuffling $D_{GAN}$ and $D_{train}$ into two equal-sized subgroups five times so that the size of each subgroup is half the size of $D_{train}$. Next, the 10 subgroups of  $D_{train}$ and $D_{GAN}$ are used to train 10 gradient-boost classifiers to label the unlabelled data and the common confident data, denoted $D_{conf}$, are then appended to $D_{train}$ and $D_{GAN}$. Instances in $D_{conf}$ are chosen by setting a threshold of the minimum number of classifiers that should agree on the predicted label of the unlabelled instance. Such threshold is set to 5. 
Other labeling methods we have tried:
 * 1gb_labelling.py
    this method traines 1GB on the subset_training instances on the VGG-16 features.
* 10gb_labelling_vgg.py   
    this method trains 10GB on the subset_training instances, features extracted 
    from  the VGG-16 network.
* dif_models_labelling.py
    This file consists of many machine learning classification algorithms. Featres are
    again extracted from VGG-16 pretrained network.
* 10gb_labelling_pca
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
We adopted the technique of Graziani et al[2] which is summarized as follows. The first step is to define the interesting concepts relevant to IDC prediction. Nuclear morphometric and appearance features such as average size and pleomorphism can help in assessing cancer grades and predicting treatment[3]. In this work, we use concepts by referring to the Nottingham Histologic Grading system (NHG)[4]. Such concepts are extracted from the nuclear segmentation dataset that quantify the impact of variations in nuclei size, area and texture. The second step is to compute the Pearson product-moment correlation coefficient $\rho$ between each concept and the network prediction for each input patch. Thd code of the global interpretability used in this work can be found on this link https://github.com/medgift/iMIMIC-RCVs.

## Other dependencies:
All the required packages should be already installed if you install nni requirements.txt in nni repo.
If you need to run interpretability, you will need you install packages mentioned in iMIMIC-RCVs/requirements.txt

## Refrences
[1] Kumar, Neeraj, Ruchika Verma, Sanuj Sharma, Surabhi Bhargava, Abhishek Vahadane, and Amit Sethi. "A dataset and a technique for generalized nuclear segmentation for computational pathology." IEEE transactions on medical imaging 36, no. 7 (2017): 1550-1560
[2] Graziani M, Andrearczyk V, Müller H. Regression concept vectors for bidirectional explanations in histopathology. InUnderstanding and Interpreting Machine Learning in Medical Image Computing Applications 2018 Sep 20 (pp. 124-132). Springer, Cham.
[3] Beck, A.H., Sangoi, A.R., Leung, S., Marinelli, R.J., Nielsen, T.O., Van De Vijver, M.J., West, R.B., Van De Rijn, M. and Koller, D., 2011. Systematic analysis of breast cancer morphology uncovers stromal features associated with survival. Science translational medicine, 3(108), pp.108ra113-108ra113.
[4] Elston, C.W. and Ellis, I.O., 1991. Pathological prognostic factors in breast cancer. I. The value of histological grade in breast cancer: experience from a large study with long‐term follow‐up. Histopathology, 19(5), pp.403-410.

    
    
