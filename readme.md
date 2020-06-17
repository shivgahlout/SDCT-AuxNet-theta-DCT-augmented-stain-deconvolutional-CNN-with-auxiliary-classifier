###Implementation of the paper SDCT-AuxNet^${\theta}$: DCT Augmented Stain Deconvolutional  CNN with Auxiliary Classifier for Cancer Diagnosis


###Requirements:
#### 1. Python 2.7
#### 2. PyTorch 0.4.1

### Dataset:
The dataset for this article is available at The Cancer Imaging Archive (TCIA): https://wiki.cancerimagingarchive.net/display/Public/C_NMC_2019+Dataset%3A+ALL+Challenge+dataset+of+ISBI+2019


### Training
1. For training the the dataset structure should have the following structure:

main_data_dir
|          -----------class1------subject_folders
|         |
fold0------
|         |
|          -----------class2------subject_folders
|  
|          -----------class1------subject_folders
|         |
fold1------
|         |
|          -----------class2------subject_folders  
.
.
.
.          -----------class1------subject_folders
.         |
fold1------
          |
           -----------class2------subject_folders

2. Run train_model.py

### Evaluation
1. Run test_model.py
2. The performance of the model on the test set can be evaluated at the challenge portal: https://competitions.codalab.org/competitions/20395
