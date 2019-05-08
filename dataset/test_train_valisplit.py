#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code organizes the downloaded dataset into the same training/testing/validation that
is provided in the baseline
"""
import os
import shutil
with open('cases_train.txt') as f:
    lines = f.read().split("\n")
    del(lines[len(lines)-1])
    training_folder='/home/ubuntu/preprocessing/maincode/files/training/'
    for i in lines:
            files_0=os.listdir('/home/ubuntu/preprocessing/dataset/' +i+'/0')
            files_1=os.listdir('/home/ubuntu/preprocessing/dataset/' +i+'/1')
            for file in files_0:
                shutil.move( '/home/ubuntu/preprocessing/dataset/' +i+'/0/'+file, training_folder)
            for file in files_1:
                shutil.move( '/home/ubuntu/preprocessing/dataset/' +i+'/1/'+file, training_folder)
with open('cases_test.txt') as f:
    lines = f.read().split("\n")
    del(lines[len(lines)-1])
    testing_folder='/home/ubuntu/preprocessing/maincode/files/testing/'
    for i in lines:
            files_0=os.listdir('/home/ubuntu/preprocessing/dataset/' +i+'/0')
            files_1=os.listdir('/home/ubuntu/preprocessing/dataset/' +i+'/1')
            for file in files_0:
                shutil.move( '/home/ubuntu/preprocessing/dataset/' +i+'/0/'+file, testing_folder)
            for file in files_1:
                shutil.move( '/home/ubuntu/preprocessing/dataset/' +i+'/1/'+file, testing_folder)
with open('cases_val.txt') as f:
    lines = f.read().split("\n")
    del(lines[len(lines)-1])
    validation_folder='/home/ubuntu/preprocessing/maincode/files/validation/'
    for i in lines:
            files_0=os.listdir('/home/ubuntu/preprocessing/dataset/' +i+'/0')
            files_1=os.listdir('/home/ubuntu/preprocessing/dataset/' +i+'/1')
            for file in files_0:
                shutil.move( '/home/ubuntu/preprocessing/dataset/' +i+'/0/'+file, validation_folder)
            for file in files_1:
                shutil.move( '/home/ubuntu/preprocessing/dataset/' +i+'/1/'+file, validation_folder)
