# -*- coding: utf-8 -*-

"""
    Data_Access
    
    Frederico Vicente, NOVA FCT, MIEI
    Ludwig Krippahl
"""

#######################################
'''           Imports               '''
#######################################

import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import imageio 
import glob
from skimage.transform import resize

#######################################
'''           Operations            '''
#######################################

def standardize(x):
    x -= 127.5
    x /= 127.5
    return x

def de_standardize(x):
    return abs((x * 127.5) + 127.5)

def de_standardize_norm(x):
    return np.interp(x,[-1,1],[0,1])

def normalize(x):
    x /= 255. # /= operator does not create temporary tensor
    return x

def de_normalize(x):
    x *= 255.
    return x

#######################################
'''           Prepare Data          '''
#######################################

def load_data(data_type = '.png', data_dir='npz_imgs', size_shape=(152,152)):
    """
    Load images from .png or .npz files into numpy vectors
    
    :data_type: .png files or .npz files
    :data_dir: directory/folder where the files are
    :size_shape: images final shape after loading (for resize purposes)
    """
    if (data_type == '.png'):
        return np.array([cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(data_dir, img)),dsize=size_shape),cv2.COLOR_BGR2RGB) for img in os.listdir(data_dir) if img.endswith(data_type)])
    elif (data_type == '.npz'):
        images = None
        max_npzs = 10
        images_counter = 0
        images_number = 0
        for img in os.listdir(data_dir):
            if(img.endswith(data_type)):
                currentSetOfImages = np.load(os.path.join(data_dir, img),'r')
                print('Loaded npz',img)
                imgs = currentSetOfImages['images'].astype(np.float32)
                if(imgs.shape[1] != size_shape[0]):
                    print('Images resized - Old shape: {} --- New shape: {}'.format(imgs.shape[1:3],size_shape))
                    imgs = np.array([resize(image,size_shape) for image in imgs])
                imgs = np.reshape(imgs,newshape=(-1,size_shape[0],size_shape[1],3))
                images_number += imgs.shape[0]
                if images is None:
                    images = imgs
                else:
                    images = np.concatenate([images,imgs],axis=0)
                images_counter += 1
            if(images_counter == max_npzs):
                break
        return images, images_number
    
def prepare_data(generator,data_dir='npz_imgs',labels_dir='npz_labels',size_shape=(152,152)):
    """
    Loads dataset info into numpy vectors and transforms the images data
    to the interval [-1,1] for GAN or [0,1] for VAE
    
    :generator: Generative model being used ('gan' or 'vae')
    :data_type: .png files or .npz files
    :data_dir: directory/folder where the files are
    :size_shape: images final shape after loading (for resize purposes)
    """
    train_x,n_images = load_data(data_type = '.npz', data_dir=data_dir, size_shape=size_shape)
    print('Data loded: ',n_images, ' images')
    
    if(generator == 'gan'):
        train_x = standardize(train_x)

    elif(generator == 'vae'):
        train_x = normalize(train_x)
        
    if labels_dir is not None:
        for label in os.listdir(labels_dir):
            currentSetOflabels = np.load(os.path.join(labels_dir, label),'r')
            labels = currentSetOflabels['labels'].astype(np.float32)
    
    return train_x, labels, None, None

def prepare_data_KERAS(generator = 'gan', dataset_name='FMNIST',limitation=10000):
    """
    Simple abstraction to easily load the standard datasets keras has builtin
    (Fashion MNIST: 'FMNIST' | MNIST: 'MNIST' | Cifar: 'cifar')
    After loading, preprocessing is done to prepare for generative models
    
    :generator: Generative model being used ('gan' or 'vae')
    :dataset_name: dataset to load (Fashion MNIST: 'FMNIST' | MNIST: 'MNIST' | Cifar: 'cifar')
    """
    
    if(dataset_name == 'FMNIST'):
        dataset = tf.keras.datasets.fashion_mnist
    elif(dataset_name == 'MNIST'):
        dataset = tf.keras.datasets.mnist
    elif(dataset_name == 'cifar'):
        dataset = tf.keras.datasets.cifar10
    else:
        print('Not supported :(')
        return None
        
    (train_x, train_y),(test_x,test_y) = dataset.load_data()
    shape_X = train_x.shape[1]
    shape_Y = train_x.shape[2]
    if(len(train_x.shape)==3):
        channels = 1
    else:
        channels = train_x.shape[3]
    
    
    if(generator == 'gan'):
        train_x = np.reshape(((train_x - 127.5) / 127.5).astype(np.float32),newshape=(-1,shape_X,shape_Y,channels))
        test_x = np.reshape(((test_x - 127.5) / 127.5).astype(np.float32),newshape=(-1,shape_X,shape_Y,channels))

    elif(generator == 'vae'):
        train_x = np.reshape((train_x / 255.).astype(np.float32),newshape=(-1,shape_X,shape_Y,channels))
        test_x = np.reshape((test_x / 255.).astype(np.float32),newshape=(-1,shape_X,shape_Y,channels))
        
        if(channels == 1):
            train_x[train_x >= .5] = 1.
            train_x[train_x < .5] = 0.
            test_x[test_x >= .5] = 1.
            test_x[test_x < .5] = 0.
        
    return train_x[:limitation],train_y[:limitation],test_x[:limitation],test_y[:limitation]
    

def prepare_dataset(generator, dataset, image_size=(152,152)):
    """
    Useful when dataset is already loaded but resize is needed or simply
    processing is needed to feed the dataset to generative models
    
    :generator: generator model working with - 'gan' or 'vae'
    :dataset: numpy array with image data
    :image_size: size of output images
    """
    train_x, train_y = dataset
    
    if(train_x.shape[1] != image_size[0]):
        print('Images resized - Old shape: {} --- New shape: {}'.format(train_x.shape[1:3],image_size))
        train_x = np.array([resize(image,image_size) for image in train_x])
    
    if(generator == 'gan'):
        train_x = standardize(train_x)
    elif(generator == 'vae'):
        train_x = normalize(train_x)
        
    return train_x,train_y,None,None

def prepare_img(img,type_de = 'gan'):
    """
    Transform tensor into storable image
    
    :img: image tensor
    :type_de: generative model being used
    """
    img = img.numpy()
    if (type_de == 'gan'):
        img = de_standardize_norm(img)
    else:
        img = de_standardize_norm(img) 
    return img

#######################################
'''      Store and load Files       '''
#######################################

def store_weights_in_file(filename,weights_):
    np.savez('{}.npz'.format(filename),weights=weights_)

def load_weights_from_file(filename):
    weights = np.load('{}.npz'.format(filename),allow_pickle=True)['weights']
    return weights

def store_seed_in_file(filename,seed_):
    np.savez('{}.npz'.format(filename),seed=seed_)

def load_seed_from_file(filename):
    weights = np.load('{}.npz'.format(filename),allow_pickle=True)['seed']
    return weights

def store_batch_norm(filename,weights_):
    if('batch_norm' not in os.listdir('./weights')):
        prepare_directory('weights/batch_norm')
    for i in range(len(weights_)):
        np.savez('weights/batch_norm/{}_{}.npz'.format(filename,i),weights=weights_[i])

def load_batch_norm(data_dir):
    ws = list()
    for weight in os.listdir(data_dir):
        currentSetOfWeights = np.load(os.path.join(data_dir, weight),'r')
        print(currentSetOfWeights.files)
        w = currentSetOfWeights['weights']
        ws.append(w)
    return ws

def write_current_epoch(filename,epoch):
    with open('{}.txt'.format(filename),'w') as ofil:
        ofil.write(f'{epoch}')
    print('Saved epoch ',epoch)
    
def write_current_phase_number(filename,index):
    with open('{}.txt'.format(filename),'w') as ofil:
        ofil.write(f'{index}')
    print('Saved phase number ',index)
    
def write_combinations(filename,phase):
    with open('{}.txt'.format(filename),'w') as f:
        print(phase, file=f)
    print('Saved training phase ')

def read_combinations(filename):
    with open('{}.txt'.format(filename),'r') as f:
        content = f.read()
        return eval(content)

def read_phase_number(filename):
    with open('{}.txt'.format(filename),'r') as f:
        return int(f.read())
        
def prepare_directory(directory = "imgs"):
    if not os.path.exists(directory):
        os.makedirs(directory)

#######################################
'''      Get CSV dataset info       '''
#######################################

def get_images_of_certain_class(csv_filename,data_dir,class_type,filter_value=1,data_type='.npz'):
    """
    Might be useful to studdy a dataset
    Gets the images that are from a certain category
    
    :csv_filename: csv file with information of dataset
    :data_dir: folder where images are (suppoted types of files: npz, png)
    :class_type: category of class
    :filter_value: extract images with a class that has this value
    :data_type: data file type (suppoted types of files: npz, png)
    """
    images,images_number = load_data(data_type = data_type, data_dir=data_dir)
    df = pd.read_csv(csv_filename,sep=',')
    df = df.head(images_number)
    class_images = df.loc[df[class_type] == filter_value]
    ix = class_images[0]
    images = images[ix]
    store_images_seed("images",images,"none")
    
def get_labels_of_certain_class(csv_filename,class_type,class_max=4,limits=(0,5000)):
    """
    Might be useful to studdy a dataset
    Gets the labels from a certain class between a range
    
    :csv_filename: csv file with information of dataset
    :class_type: category of class
    :filter_value: extract images with a class that has this value
    :data_type: data file type (suppoted types of files: npz, png)
    """
    #csv_filename = r"C:\Users\MrVicente\Desktop\trains.csv"
    df = pd.read_csv(csv_filename,sep=',')
    df = df[limits[0]:limits[1]]
    class_labels = df[class_type]
    labels = list()
    for class_value in class_labels:
        one_hot = np.zeros(class_max)
        one_hot[class_value-1] = 1
        labels.append(one_hot)
    np.savez('{}_{}.npz'.format('labels',class_type),labels=labels)

#######################################
'''          Store images           '''
#######################################

def store_image_simple(directory,image_name,image,prediction):
    plt.axis('off')
    plt.imsave('{}/{}.png'.format(directory,image_name),image)
    
def store_image(directory,epoch,image,i,type_de):
    prepare_directory(directory)
    prepare_directory('{}/epoch_{}'.format(directory,epoch))
    plt.axis('off')
    plt.xlabel(epoch)
    image = prepare_img(image,type_de)
    if(image.shape[-1] != 1):
        plt.imsave('{}/epoch_{}/id_{}.png'.format(directory,epoch,i),image)
    else:
        plt.imsave('{}/epoch_{}/id_{}.png'.format(directory,epoch,i),image[:,:,0],cmap="gray")

def store_images_seed(directory,images,epoch,type_de='gan'):
    for i in range(len(images)):
        store_image(directory,epoch,images[i],i,type_de)


def tf_store_image(image,epoch,i):
    img = tf.image.convert_image_dtype(image,tf.uint8)
    img = tf.image.encode_jpeg(img,quality=100)
    folder = 'step{}'.format(epoch)
    prepare_directory(folder)
    dirt= '{}/test{}.jpg'.format(folder,i)
    tf.io.write_file(dirt,img)

def store_images(images,epoch):
    plt.figure(figsize=(2,2))
    for i in range(images.shape[0]):
        plt.subplot(2, 2, i+1)
        image = prepare_img(images[i])
        plt.imshow(image)
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

def create_collection(epochs,n_gen_images,directory):
    """
    Creates a png image with the evolution of each seed image thoughout the training process
    
    :epochs: number of epochs the model trained
    :n_gen_images: number of images generated per epoch
    :directory: folder to store png image
    """
    fig = plt.figure(figsize=(20,200))
    k=0
    for epoch in range(epochs):
        for im in range(n_gen_images):
            k+=1
            img = mpimg.imread('{}/epoch_{}/id_{}.png'.format(directory,epoch,im))
            fig.add_subplot(epochs, n_gen_images, k)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
    plt.tight_layout()
    plt.savefig('training.png')
    
#######################################
'''         Visualizations          '''
#######################################

def plot_image(i, predictions_array, images,classes=[""],directory='abc'):
    """
    Store Generated Image and plot it with the correspondent classification associated
    to it
    
    :i: id of image
    :predictions_array: array with probability for each class
    :image_size: size of output images
    :images: images generated
    :classes: Array with label names of each class
    :directory: Directory to store images
    """
    img = prepare_img(images[i])
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    if(img.shape[-1] != 1):
        plt.imshow(img)
        plt.imsave('{}/id_{}.png'.format(directory,i),img)
    else:
        plt.imshow(img[:,:,0],cmap="gray")
        plt.imsave('{}/id_{}.png'.format(directory,i),img[:,:,0],cmap="gray")

    predicted_label = np.argmax(predictions_array)

    plt.xlabel("{} {:2.0f}%".format(classes[predicted_label],
                                100*np.max(predictions_array)),
                                color='red')

def produce_generate_figure(directory,gen_images,predictions,classes=[""]):
    """
    Store Generated Images and plot them with the correspondent classifications associated
    to the images
    
    :gen_images: images generated
    :predictions: array with collection of predictions (one collection for each image)
    :classes: Array with label names of each class
    :directory: Directory to store images
    """
    num_classes = len(classes)
    prepare_directory(directory)
    num_images = gen_images.shape[0]
    num_cols = 2
    num_rows = num_images/num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    basis = tf.convert_to_tensor([0,1],dtype=tf.float32)
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        if(num_classes == 2):
            x = tf.subtract(predictions[i],basis)
            w_list = tf.abs(x)
        else:
            w_list = predictions[i]
        w_list = tf.reshape(w_list,(w_list.shape[0],))
        plot_image(i, w_list, gen_images,classes,directory)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, w_list, classes)
    plt.tight_layout()
    plt.savefig('{}/classifications.png'.format(directory))

def plot_value_array(i, predictions_array,classes = [""]):
    """
    Plot the classification prediction of each image
    :i: id of image
    :predictions_array: array with probability for each class
    :classes: Array with label names of each class
    """
    plt.grid(False)
    plt.xticks(range(len(classes)))
    plt.yticks([])
    thisplot = plt.bar(range(len(classes)), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    _ = plt.xticks(range(len(classes)),classes,rotation=80)
  
def create_gif(filename):
    """
    Create a gif from a set of images
    :filename: name of gif file
    """
    anim_file = '{}.gif'.format(filename)
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob('image*.png')
        filenames = sorted(filenames)
        last = -1
        for i,filename in enumerate(filenames):
            frame = 2*(i**0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)

    try:
        from google.colab import files
    except ImportError:
        pass
    else:
        files.download(anim_file)


#######################################
'''             Stats              '''
#######################################

def stats(predictions):
    counter = 0
    n_images = len(predictions)
    for i in range(n_images):
        t = predictions[i]
        if t > 0.50:
            counter += 1
    print('From {} images, {} belong to class C ({}%)'.format(n_images,counter,(counter/n_images) * 100))
            
def train_percentage(data_dir='npz_imgs',size_shape=(152,152),threshold = 65):
    train_x,npzs = load_data(data_type = '.npz', data_dir=data_dir, size_shape=size_shape)
    print('Data loded: ',npzs,' npz files - ',npzs * 5000, ' images')
    total_pixels = size_shape[0] * size_shape[1]
    percentages = list()
    for image in train_x:
        train_pixel = 0
        for x in range(size_shape[0]):
            for y in range(size_shape[1]):
                pixel = image[x,y]
                if(pixel[0] < threshold and pixel[1] < threshold and pixel[2] < threshold ):
                    train_pixel += 1
        percentage = (train_pixel / total_pixels) * 100
        percentages.append(percentage)
    avg_percentage = np.average(np.asarray(percentage))
    print('Avg amount of train pixels -> {} / {}'.format(avg_percentage,total_pixels))
    
#######################################
'''             Prints              '''
#######################################
        
def print_training_output(step, n_steps, critic_loss, gen_loss):
    print('-----------------------------')
    print('{}th OUT OF {} steps'.format(step,n_steps))
    print('Critic Loss: {}'.format(critic_loss))
    print('Generator Loss: {}'.format(gen_loss))
    
def print_training_output_vae(step,steps,inf_loss,gen_loss):
     print('Step {} of {}'.format(step,steps))
     print('Inference Loss: {} ------ Generator Loss: {}'.format(inf_loss,gen_loss))
     print('-------------------------------------------------')
     
def print_training_output_simple_loss(step,steps,loss):
     print('Step {} of {}'.format(step,steps))
     print('Model loss: {}'.format(loss))
     print('-------------------------------------------------')
     
def print_training_time(start_time,end_time,params):
    total_minutes = (end_time-start_time) / 60
    hours = (int)(total_minutes // 60)
    minutes = (int)(((total_minutes / 60) % 1) * 60)
    if(params is not None):
        print('Training model took: {}h and {}m with \n params: {}'.format(hours,minutes,{h: params[h] for h in params}))
    else:
        print('Training model took: {}h and {}m'.format(hours,minutes))