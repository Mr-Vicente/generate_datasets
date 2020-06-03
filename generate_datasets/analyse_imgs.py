# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:40:44 2020

@author: MrVicente
"""
import argparse

import tensorflow as tf
import tkinter as tk
from PIL import ImageTk
import os
import numpy as np
import cv2
import data_access


class ManualClassifier:
    def __init__(self,root=None,classifier_name='./classifiers/hair_classifier_v09933.h5',size_shape=(128,128), \
                 class_names=['Blond-yellow','Yellow','Orange','Orange-Brown','Blond','Light-Brown','Brown','Black','Gray','White'], \
                 images_path = r"C:\Users\MrVicente\Desktop\train_images_sample" , attribute= 'hair_color',dataset_name='dataset'):
        canvas = tk.Canvas(root,width=300,height=300)
        canvas.pack()
        
        self.root = root
        self.classifier = tf.keras.models.load_model(classifier_name)
        self.size_shape = size_shape
        self.class_names = class_names
        self.images_path = images_path
        self.attribute = attribute
        self.dataset_name = dataset_name
        
        self.real_images = list()
        self.images_to_store = list()
        self.labels = list()
        self.ImageTk_images = list()
        self.index = 0
        
        self.load_images_predict()
        
    def load_images_predict(self):
        for filename in os.listdir(self.images_path):
            imagePath = self.images_path + "\\" + filename
            image = np.array(cv2.cvtColor(cv2.resize(cv2.imread(imagePath),dsize=(self.size_shape[0],self.size_shape[1])),cv2.COLOR_BGR2RGB))
            self.real_images.append(image)
            img = ImageTk.PhotoImage(file=imagePath)
            self.ImageTk_images.append(img)

        self.real_images = np.array(self.real_images).astype(np.float32)
        images_norms = data_access.normalize(data_access.de_standardize(self.real_images))
        self.predictions = self.classifier(images_norms).numpy()
        
    def changeImage(self,n_class,class_max):
        one_hot = np.zeros(class_max)
        one_hot[n_class] = 1
        print(one_hot)
        if(self.index < len(self.real_images)-1):
            prediction_i = np.argmax(self.predictions[self.index])
            if(n_class != prediction_i):
                self.labels.append(one_hot)
                self.images_to_store.append(self.real_images[self.index])
            self.index += 1
            prediction_i = np.argmax(self.predictions[self.index])
            self.panel.configure(image=self.ImageTk_images[self.index],text = 'Predicted: ' + self.class_names[prediction_i])
        else:
            ls = np.array(self.labels).astype(np.float32)
            self.images_to_store = np.array(self.images_to_store).astype(np.float32)
            print('labels to store: ', ls.shape)
            print('imgs to store: ', self.images_to_store.shape)
            np.savez('{}_images_{}_{}x{}_{}.npz'.format(self.dataset_name,self.attribute,self.size_shape[0],self.size_shape[1],'cc'),images=self.images_to_store)
            np.savez('{}_labels_{}_{}x{}_{}.npz'.format(self.dataset_name,self.attribute,self.size_shape[0],self.size_shape[1],'cc'),labels=ls)


    def draw_buttons(self):
        for i,c_name in enumerate(self.class_names):
            button = tk.Button(self.root, text = c_name, command = lambda n_class = i,class_max=len(self.class_names): self.changeImage(n_class,class_max), width = 20, justify='center')
            button.place(x = 50,y = i * 30 + 10)


    def draw_gui(self):
        prediction_i = np.argmax(self.predictions[self.index])
        self.panel = tk.Label(self.root, image= self.ImageTk_images[self.index],text = 'Predicted: ' + self.class_names[prediction_i], compound=tk.RIGHT)
        self.panel.pack(side="right", fill="both", expand="yes")
        
        self.draw_buttons()

        
def main(args):
    
    root = tk.Tk()
    root.wm_title("Manual Classifier")
    class_names = ['1 waggon', '2 waggons', '3 waggons', '4 waggons']
    path= r"C:\Users\MrVicente\Desktop\train_images"
    gui = ManualClassifier(root,args.classifier_name,args.shape_size, class_names = class_names,images_path=path,attribute='n-waggons')
    gui.draw_gui()
    root.mainloop()
    #'./classifiers/hair_classifier_v09933.h5'

if __name__ == '__main__':
    
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--classifier_name', type=str, default='./classifiers/Number_Of_Wagons_0.9952_ConvExp14_t4000_v1000.hdf5', help='saved classifier model filename')
    arg_parser.add_argument('--shape_size', type=tuple, default=(152,152), help='batch size')

    args = arg_parser.parse_args()
    main(args)