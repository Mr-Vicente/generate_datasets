
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 09:19:04 2020

@author: MrVicente
        This snippet of code was based on tkarras PGGAN repo,
        https://github.com/tkarras/progressive_growing_of_gans/blob/master/dataset_tool.py
        
        To download the dataset go to this repo: https://github.com/fyu/lsun
        Run the download.py file and you have the dataset.
        
        However the dataset is in lmdb format so you will have to preprocess it to work with the images.
        This code turns the images in lmdb format into a rgb numpy array, and also
        partitions the dataset in multiple npz files -- set n_images_per_partition = max_images
        if you only want one npz file
"""

from __future__ import print_function
import lmdb # pip install lmdb
import cv2 # pip install opencv-python
import io
import PIL.Image
import numpy as np
import argparse

def create_lsun(lmdb_dir, resolution=256, max_images=None,n_images_per_partition=5000):
    print('Loading LSUN Bedroom dataset from "%s"' % lmdb_dir)
    with lmdb.open(lmdb_dir, readonly=True).begin(write=False) as txn:
        total_images = txn.stat()['entries']
        print('Getting {} images out of {} images'.format(max_images, total_images))
        counter = 0
        images = list()
        for idx, (key, value) in enumerate(txn.cursor()):
            try:
                try:
                    img = cv2.imdecode(np.fromstring(value, dtype=np.uint8), 1)
                    if img is None:
                        raise IOError('cv2.imdecode failed')
                    img = img[:, :, ::-1] # BGR => RGB
                except IOError:
                    img = np.asarray(PIL.Image.open(io.BytesIO(value)))
                crop = np.min(img.shape[:2])
                img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
                img = PIL.Image.fromarray(img, 'RGB')
                img = img.resize((resolution, resolution), PIL.Image.ANTIALIAS)
                img = np.asarray(img)
                #img = img.transpose(2, 0, 1) # HWC => CHW
                images.append(img)
                counter += 1
            except:
                print("Error")
            if (counter != 0) and ((counter % n_images_per_partition) == 0):
                images = np.asarray(images)
                print('Images shape: {}'.format(images.shape))
                filename = 'lsun_bedroom_{}.npz'.format(int(counter/5000))
                np.savez(filename,images=images)
                print('Saved 5000 images in npz format -- with filename: {}'.format(n_images_per_partition,filename))
                images = list()
            if counter == max_images:
                break
    print('Exited lsun processing')

def main(args):
    print("Started processing")
    create_lsun(args.lmdb_dir, resolution=args.resolution, max_images=args.max_images,n_images_per_partition=args.n_images_per_partition)

if __name__ == '__main__':
    
    arg_parser = argparse.ArgumentParser()
    arg_parser.add('--lmdb_dir', type=str, default='bedroom_train_lmdb', help='Folder/Directory name inside the bedroom_train_lmdb.zip')
    arg_parser.add('--resolution', type=int, default=256, help='Size of images')
    arg_parser.add('--max_images', type=int, default=10000, help='Number of images wanted (The dataset is pretty big :)')
    arg_parser.add('--n_images_per_partition', type=int, default=5000, help='Number of images per npz file')

    args = arg_parser.parse_args()
    main(args)
