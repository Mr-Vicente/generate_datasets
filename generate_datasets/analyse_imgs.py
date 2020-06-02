# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:40:44 2020

@author: MrVicente
"""

import tensorflow as tf
import tkinter as tk
from PIL import ImageTk
import os
import numpy as np
import cv2
import data_access

class_names = ['Blond-yellow','Yellow','Orange','Orange-Brown','Blond','Light-Brown','Brown','Black','Gray','White']
path= r"C:\Users\MrVicente\Desktop\train_images_sample"
index = 0

root = tk.Tk()
root.wm_title("Manually classify")

canvas = tk.Canvas(root,width=300,height=300)
canvas.pack()

imagePath = path + "\\" + os.listdir(path)[index]
print(imagePath)
img = ImageTk.PhotoImage(file=imagePath)

images = list()
classifier = tf.keras.models.load_model('./classifiers/hair_classifier_v09933.h5')

ims = list()
size_shape = (128,128)
attribute = 'hair_color'
for filename in os.listdir(path):
    imagePath = path + "\\" + filename
    image = np.array(cv2.cvtColor(cv2.resize(cv2.imread(imagePath),dsize=(size_shape[0],size_shape[1])),cv2.COLOR_BGR2RGB))
    ims.append(image)
    img = ImageTk.PhotoImage(file=imagePath)
    images.append(img)

ims = np.array(ims).astype(np.float32)
images_norms = data_access.normalize(data_access.de_standardize(ims))
predictions = classifier(images_norms).numpy()

prediction_i = np.argmax(predictions[index])
panel = tk.Label(root, image = images[index],text = class_names[prediction_i], compound=tk.RIGHT)
panel.pack(side="right", fill="both", expand="yes")

labels = list()
def changeImage(n_class,class_max):
    one_hot = np.zeros(class_max)
    one_hot[n_class] = 1
    labels.append(one_hot)
    print(one_hot)
    global index
    if(index < len(images)-1):
        index += 1
        prediction_i = np.argmax(predictions[index])
        panel.configure(image=images[index],text = class_names[prediction_i])
    else:
        ls = np.array(labels)
        np.savez('{}_{}_{}x{}_{}.npz'.format('cartoon_images',attribute,size_shape[0],size_shape[1],'cc'),images=ims)
        np.savez('{}_{}_{}x{}_{}.npz'.format('cartoon_labels',attribute,size_shape[0],size_shape[1],'cc'),labels=ls)

for i,c_name in enumerate(class_names):
    button = tk.Button(root, text = c_name, command = lambda n_class = i,class_max=len(class_names): changeImage(n_class,class_max), width = 20, justify='center')
    button.place(x = 50,y = i * 30 + 10)

root.mainloop()