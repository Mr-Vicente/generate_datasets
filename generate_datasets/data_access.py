# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 08:37:38 2020
"""

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

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class_names = ['TypeC','TypeO']

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
    x /= 255.
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
    """
    if (data_type == '.png'):
        return np.array([cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(data_dir, img)),dsize=size_shape),cv2.COLOR_BGR2RGB) for img in os.listdir(data_dir) if img.endswith(data_type)])
    elif (data_type == '.npz'):
        images = None
        max_npzs = 10
        images_counter = 0
        for img in os.listdir(data_dir):
            if(img.endswith(data_type)):
                currentSetOfImages = np.load(os.path.join(data_dir, img),'r')
                print('Loaded npz',img)
                imgs = currentSetOfImages['images'].astype(np.float32)
                if(imgs.shape[1] != size_shape[0]):
                    print('Images resized - Old shape: {} --- New shape: {}'.format(imgs.shape[1:3],size_shape))
                    imgs = np.array([resize(image,size_shape) for image in imgs])
                imgs = np.reshape(imgs,newshape=(-1,size_shape[0],size_shape[1],3))
                if images is None:
                    images = imgs
                else:
                    images = np.concatenate([images,imgs],axis=0)
                images_counter += 1
            if(images_counter == max_npzs):
                break
        return images,images_counter
    
def prepare_data(generator, batch_size = 1,data_dir='npz_imgs',size_shape=(152,152)):

    train_x,npzs = load_data(data_type = '.npz', data_dir=data_dir, size_shape=size_shape)
    print('Data loded: ',npzs,' npz files - ',npzs * 5000, ' images')
    
    if(generator == 'gan'):
        train_x = standardize(train_x)

    elif(generator == 'vae'):
        train_x = normalize(train_x)
        #buffer_size_train = train_x.shape[0]
        #train_x = tf.data.Dataset.from_tensor_slices(train_x).shuffle(buffer_size_train).batch(batch_size,drop_remainder=True)

    return train_x,None, None, None

def prepare_dataset(generator, dataset, image_size=(152,152)):
    """
    :generator: generator model working with - <gan> or <vae>
    :dataset: numpy array with image data
    :image_size: size of output images
    """
    if(dataset.shape[1] != image_size[0]):
        print('Images resized - Old shape: {} --- New shape: {}'.format(dataset.shape[1:3],image_size))
        train_x = np.array([resize(dataset,image_size) for image in dataset])
    
    if(generator == 'gan'):
        train_x = standardize(train_x)
    elif(generator == 'vae'):
        train_x = normalize(train_x)
        
    return train_x,None,None,None

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

#######################################
'''      Store and load Images      '''
#######################################

def get_images_of_certain_class(csv_filename,data_dir,class_type):
    images,npzs = load_data(data_type = '.npz', data_dir=data_dir)
    images_size = 5000 * npzs
    images = tf.convert_to_tensor(images,dtype=tf.float32)
    images = tf.reshape(images,shape=(images_size,152,152,3))
    df = pd.read_csv(csv_filename,sep=',')
    #class_type = TypeA (for example)
    df = df.head(images_size)
    class_images = df.loc[df[class_type] == 1]
    ix = class_images['name']
    print(ix)
    images = tf.gather(images,ix) #images[ix]
    store_images_seed("images",images,"none")


def prepare_img(img,type_de = 'gan'):
    img = img.numpy()
    if (type_de == 'gan'):
        img = de_standardize_norm(img)
    else:
        pass
    return img

def store_image_simple(directory,image_name,image,prediction):

    plt.axis('off')
    plt.imsave('{}/{}.png'.format(directory,image_name),image)
    
    
def store_image(directory,epoch,image,i,type_de):
    prepare_directory(directory)
    prepare_directory('{}/epoch_{}'.format(directory,epoch))
    plt.axis('off')
    plt.xlabel(epoch)
    image = prepare_img(image,type_de)

    plt.imsave('{}/epoch_{}/id_{}.png'.format(directory,epoch,i),image)

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
    fig = plt.figure(figsize=(2,2))
    for i in range(images.shape[0]):
        plt.subplot(2, 2, i+1)
        image = prepare_img(images[i])
        plt.imshow(image)
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

def create_collection(epoches,n_dif_images,directory):
    fig = plt.figure(figsize=(20,200))
    k=0
    for epoch in range(epoches):
        for im in range(n_dif_images):
            k+=1
            img = mpimg.imread('{}/epoch_{}/id_{}.png'.format(directory,epoch,im))
            fig.add_subplot(epoches, n_dif_images, k)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
    plt.tight_layout()
    plt.savefig('training.png')
    
def write_current_epoch(filename,epoch):
    with open('{}.txt'.format(filename),'w') as ofil:
        ofil.write(f'{epoch}')
    print('Saved epoch ',epoch)
        
def prepare_directory(directory = "imgs"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
#######################################
'''         Visualizations          '''
#######################################

def plot_image(i, predictions_array, images):
    img = prepare_img(images[i])
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img)

    predicted_label = np.argmax(predictions_array)

    plt.xlabel("{} {:2.0f}%".format(class_names[predicted_label],
                                100*np.max(predictions_array)),
                                color='red')

def produce_generate_figure(directory,gen_images,predictions):
    prepare_directory(directory)
    num_images = gen_images.shape[0]
    num_cols = 2
    num_rows = num_images/num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    basis = tf.convert_to_tensor([0,1],dtype=tf.float32)
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        p = tf.subtract(predictions[i],basis)
        p = tf.abs(p)
        p = tf.reshape(p,shape=(2,))
        plot_image(i, p, gen_images)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, p)
        _ = plt.xticks(range(len(class_names)),class_names,rotation=80)
    plt.tight_layout()
    plt.savefig('{}/classifications.png'.format(directory))

def plot_value_array(i, predictions_array):
    plt.grid(False)
    plt.xticks(range(len(class_names)))
    plt.yticks([])
    thisplot = plt.bar(range(len(class_names)), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
  
def create_gif(filename):
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
        