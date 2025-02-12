import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import copy
from matplotlib.colors import Normalize


#Hugging face
import datasets

import pdb


def scale_to_range_neg1_to_1(np_array):
    """
    Scales the input tensor to the range (-1, 1).
    
    Args:
        tensor (torch.Tensor): Input tensor to scale.

    Returns:
        torch.Tensor: Tensor scaled to the range (-1, 1).
    """
    min_val = np.min(np_array)
    max_val = np.max(np_array)

    if min_val == max_val:
        return np_array
    
    # Scale to range [0, 1]
    scaled_tensor = (np_array - min_val) / (max_val - min_val)
    
    # Scale to range [-1, 1]
    scaled_tensor = scaled_tensor * 2 - 1
    
    return scaled_tensor



def save_image_with_landmarks(image, landmarks, filename):
    """
    Save an image with landmarks overlaid as dots for visualization.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap="gray")
    for (x, y) in landmarks:
        plt.scatter(x, y, c="red", s=20)  # Mark landmarks with red dots
    plt.axis("off")
    plt.colorbar()
    plt.savefig(filename)
    plt.close()

class PelvisXRayDataset(datasets.Dataset):

    def __init__(self, arrow_table, csv_file, transform=None, train=True):
        """
        Arguments:
            csv_file (string): Path to the csv file with landmark annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(arrow_table)
        self.arrow_table = arrow_table
        self.landmarks_df = pd.read_csv(csv_file)
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.arrow_table)

    def __getitem__(self, idx):
        #pdb.set_trace()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.train:
            data_folder = 'DATA_FOLDER_npy/train'
            mask_folder = 'MASK_FOLDER_npy/all/train'
        else:
            data_folder = 'DATA_FOLDER_npy/test'
            mask_folder = 'MASK_FOLDER_npy/all/test'

        #pdb.set_trace()
        
        # idx is [127, 187, 138]
        data_files = os.listdir(data_folder)
        mask_files = os.listdir(mask_folder)
        #batch = super().__getitem__(idx) 
        batch = {}
        # X-ray images are (0, 1)
        images_list = []
        seg_all_list = []
        image_filenames_list = []
        for i in idx:
            np_data = np.load(os.path.join(data_folder, data_files[i]))[[0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], :, :]
            for j in range(np_data.shape[0]):
                np_data[j] = scale_to_range_neg1_to_1(np_data[j])
                #pdb.set_trace()
            images_list.append(np_data)
            seg_all_list.append(np.expand_dims(np.load(os.path.join(mask_folder, data_files[i]))[0,:,:], axis=0))
            image_filenames_list.append(data_files[i])
        #images_list = [np.load('DATA_FOLDER/train/{}.npy'.format(idx[0])), np.load('DATA_FOLDER/train/{}.npy'.format(idx[1])), np.load('DATA_FOLDER/train/{}.npy'.format(idx[2]))]
        #seg_all_list = [np.load('MASK_FOLDER/all/train/{}.npy'.format(idx[0])), np.load('MASK_FOLDER/all/train/{}.npy'.format(idx[1])), np.load('MASK_FOLDER/all/train/{}.npy'.format(idx[2]))]
        #image_filenames_list = ['{}.npy'.format(idx[0]), '{}.npy'.format(idx[1]), '{}.npy'.format(idx[2])]
        
        images_tensor = torch.tensor(np.array(images_list))

        #plt.imshow(images_tensor[0][1], cmap="gray")
        #plt.savefig("data vis")

        seg_tensor = torch.tensor(np.array(seg_all_list)) 

        batch['images'] = images_tensor
        batch['seg_all'] = seg_tensor
        batch['image_filenames'] = image_filenames_list


        #batch = super().__getitem__(idx) 
        #batch is a dictionary with ['images'] containing a list of 3 (1, 384, 384) images
        # and with ['seg_all'] containing 3 (1, 384, 384) landmark locations 
        # Potential error?: landmark (1, 384, 384) tensor is rescaled to have values ranging from 0 to 0.0549
        # and with ['image_filenames'] containing 3 file names

        return {
            'images':batch['images'],
            'seg_all':batch['seg_all'], 
            'image_filenames':batch['image_filenames']
        }