# -*- coding: utf-8 -*-
"""
Created on Sun May 24 21:55:05 2020

@author: MrVicente
"""
from __future__ import print_function
import cv2 # pip install opencv-python
import numpy as np
import os
import pandas as pd

def encode_data_cartoon(path_to_local='D:\Experiments',data_dir='cartoonset10k', size_shape=(128,128), attribute="hair_color"):
    """
    Load images from .png or .npz files into numpy vectors
    """
    images = list()
    labels = list()
    for filename in os.listdir(data_dir):
        if(filename.endswith('.png')):
            print(filename)
            img = cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(data_dir, filename)),dsize=size_shape),cv2.COLOR_BGR2RGB) 
            images.append(img)
        elif(filename.endswith('.csv')):
            print(filename)
            PATH = r"{}\{}\{}".format(path_to_local,data_dir,filename)
            imageInfo = pd.read_csv(PATH,header=None,sep=',',encoding='utf-8')
            attr = imageInfo.loc[imageInfo[0] == attribute]
            class_n = attr[1].values[0]
            class_max = attr[2].values[0]
            one_hot = np.zeros(class_max)
            one_hot[class_n] = 1
            labels.append(one_hot)
           
    images = np.array(images)
    labels = np.array(labels)
    size = images.shape[0]
    images = np.split(images,size // 5000)
    labels = np.split(labels,size // 5000)
    for i in range(len(images)):
        np.savez('{}_{}_{}x{}_{}.npz'.format('cartoon_images',attribute,size_shape[0],size_shape[1],i),images=images[i])
        np.savez('{}_{}_{}x{}_{}.npz'.format('cartoon_labels',attribute,size_shape[0],size_shape[1],i),labels=labels[i])

def decode_data_cartoon(data_dir='dataset_info/',size_shape=(128,128)):
    images_vec = None
    labels_vec = None
    max_npzs = 10
    images_counter = 0
    labels_counter = 0
    folder = data_dir + 'cartoon_images'
    for images in sorted(os.listdir(folder)):
        print(images)
        currentSetOfImages = np.load(os.path.join(folder, images),'r')
        imgs = currentSetOfImages['images'].astype(np.float32)
        imgs = np.reshape(imgs,newshape=(-1,size_shape[0],size_shape[1],3))
        if images_vec is None:
            images_vec = imgs
        else:
            images_vec = np.concatenate([images_vec,imgs],axis=0)
        images_counter += 1
        if(images_counter == max_npzs):
            break
    folder = data_dir + 'cartoon_labels'
    for labelsSet in sorted(os.listdir(folder)):
        print(labelsSet)
        currentSetOflabels = np.load(os.path.join(folder, labelsSet),'r')
        labels_ = currentSetOflabels['labels'].astype(np.float32)
        labels_ = np.reshape(labels_,newshape=(-1,labels_.shape[1]))
        if labels_vec is None:
            labels_vec = labels_
        else:
            labels_vec = np.concatenate([labels_vec,labels_],axis=0)
        labels_counter += 1
        if(labels_counter == max_npzs):
            break
        
    return images_vec,labels_vec
