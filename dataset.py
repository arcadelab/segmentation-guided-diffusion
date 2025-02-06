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


'''
def get_box(x):
    box_radius = 16
    return [(int(x[0]) - box_radius, int(x[1]) - box_radius),
        (int(x[0]) + box_radius, int(x[1]) + box_radius)]

def draw_land(image_array, x, number):
    # Get bounding box for the landmark
    bbox = get_box(x)
    x_min, y_min = bbox[0]
    x_max, y_max = bbox[1]

    # Draw a circle filled with the `number` value
    for i in range(y_min, y_max):
        for j in range(x_min, x_max):
            # Ensure we are within the circle and bounds
            if ((j - x[0])**2 + (i - x[1])**2) <= 16**2:
                if 0 <= i < image_array.shape[1] and 0 <= j < image_array.shape[2] and image_array[0][int(i)][int(j)] == 0:
                    image_array[0][int(i)][int(j)] = number

"""Converts a list of x and y keypoint coordinate lists into a numpy array representing 
an image of the landmarks"""
def landmark_img_from_keypoints(transformed_keypoints, img_dim):
    land_proj = torch.zeros(1, img_dim, img_dim)
    for land_idx, cur_land in enumerate(transformed_keypoints):
        if (cur_land[0] >= 0) and (cur_land[1] >= 0) and \
                            (cur_land[0] < img_dim) and (cur_land[1] < img_dim):
            draw_land(land_proj, cur_land, land_idx + 1) 
    return land_proj
'''
def get_box(x, radius):
    """Get bounding box coordinates for a circle of a given radius."""
    return [(int(x[0]) - radius, int(x[1]) - radius),
            (int(x[0]) + radius, int(x[1]) + radius)]

def draw_land_with_boundaries(image_array, x, number, radius, centers):
    """
    Draw a filled circle while respecting boundaries with other centers.
    Assign pixels based on proximity to the current center vs others.
    """
    bbox = get_box(x, radius)
    x_min, y_min = bbox[0]
    x_max, y_max = bbox[1]

    updated = False
    for i in range(y_min, y_max):
        for j in range(x_min, x_max):
            if 0 <= i < image_array.shape[1] and 0 <= j < image_array.shape[2]:
                if ((j - x[0])**2 + (i - x[1])**2) <= radius**2:
                    # Calculate distances to all centers
                    distances = [((j - cx)**2 + (i - cy)**2) for cx, cy in centers]
                    closest_center = np.argmin(distances)
                    
                    # If this landmark is the closest, claim the pixel
                    if closest_center == number - 1 and image_array[0][i][j] == 0:
                        image_array[0][i][j] = number
                        updated = True
    return updated

def landmark_img_from_keypoints_with_boundaries(transformed_keypoints, img_dim, max_radius, step=1):
    """Generate landmark image with shared boundaries for close circles."""
    land_proj = torch.zeros(1, img_dim, img_dim)
    radii = [1] * len(transformed_keypoints)  # Initial radii of circles
    done = [False] * len(transformed_keypoints)  # Track if a landmark is done expanding

    while not all(done):
        for land_idx, cur_land in enumerate(transformed_keypoints):
            if done[land_idx]:
                continue  # Skip landmarks that are done expanding
            
            # Try to expand the current landmark while respecting boundaries
            expanded = draw_land_with_boundaries(
                land_proj, cur_land, land_idx + 1, radii[land_idx] + step, transformed_keypoints
            )
            if expanded:
                radii[land_idx] += step  # Increment radius if expansion succeeded
            else:
                done[land_idx] = True  # Mark as done if no further expansion is possible
            
            # Stop expansion if the maximum radius is reached
            if radii[land_idx] >= max_radius:
                done[land_idx] = True

    return land_proj


class PelvisXRayDataset(datasets.Dataset):

    def __init__(self, arrow_table, csv_file, transform=None):
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

    def __len__(self):
        return len(self.arrow_table)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_folder = 'DATA_FOLDER/train'
        mask_folder = 'MASK_FOLDER/all/train'

        # idx is [127, 187, 138]
        data_files = os.listdir(data_folder)
        mask_files = os.listdir(mask_folder)

        #batch = super().__getitem__(idx) 
        batch = {}
        # X-ray images are (0, 1)
        images_list = [np.expand_dims(np.load(os.path.join(data_folder, data_files[i]))[0,:,:], axis=0) for i in idx]
        seg_all_list = [np.expand_dims(np.load(os.path.join(mask_folder, data_files[i]))[0,:,:], axis=0) for i in idx]
        image_filenames_list = [data_files[i] for i in idx]
        #images_list = [np.load('DATA_FOLDER/train/{}.npy'.format(idx[0])), np.load('DATA_FOLDER/train/{}.npy'.format(idx[1])), np.load('DATA_FOLDER/train/{}.npy'.format(idx[2]))]
        #seg_all_list = [np.load('MASK_FOLDER/all/train/{}.npy'.format(idx[0])), np.load('MASK_FOLDER/all/train/{}.npy'.format(idx[1])), np.load('MASK_FOLDER/all/train/{}.npy'.format(idx[2]))]
        #image_filenames_list = ['{}.npy'.format(idx[0]), '{}.npy'.format(idx[1]), '{}.npy'.format(idx[2])]

        batch['images'] = images_list
        batch['seg_all'] = seg_all_list
        batch['image_filenames'] = image_filenames_list

        '''
        if self.transform:
            for i in range(len(batch['image_filenames'])):
                #original_filename = os.path.join(output_dir, f"original_image_{i}.png")
                #save_image_with_landmarks(batch['images'][i].squeeze(0), landmarks[i], original_filename)
                
                np_img = np.stack([np.array(batch['images'][i]).squeeze(0)] * 3, axis=-1) # reshape image to (384, 384, 3)
                #np_img = np.array(batch['images'][i])
                np_img = (((np_img + 1)/2)*255).astype(np.uint8) # Normalize to (0, 225)

                transformed = self.transform(image=np_img)
                transformed_image = transformed['image']
                

                # Reshape transformed_image from (384, 384, 3) to (1, 384, 384) 
                transformed_image = np.asarray(transformed_image)[:, :, 0]                
                # Reshape to (1, 384, 384)
                transformed_image = transformed_image[np.newaxis, :, :]

                #Update transformed_batch
                batch['images'][i] = transformed_image


        #batch is a dictionary with ['images'] containing a list of 3 (1, 384, 384) images
        # and with ['seg_all'] containing 3 (1, 384, 384) landmark locations 
        # Potential error?: landmark (1, 384, 384) tensor is rescaled to have values ranging from 0 to 0.0549
        # and with ['image_filenames'] containing 3 file names

        landmarks = []
        for image_filename in batch['image_filenames']:
            idx = int(image_filename.split(".")[0])
            landmarks.append(
                list([self.landmarks_df.iloc[idx, 1 + i], self.landmarks_df.iloc[idx, 1 + i + 1]]
                for i in range(0, 28, 2))
            )


        #transformed_batch = copy.deepcopy(batch) #This is inefficient and for debugging (optimize later)
        #output_dir = "debug_images"
        #os.makedirs(output_dir, exist_ok=True)  # Create output directory if not exists
        
        if self.transform:
            for i in range(len(batch['image_filenames'])):
                #original_filename = os.path.join(output_dir, f"original_image_{i}.png")
                #save_image_with_landmarks(batch['images'][i].squeeze(0), landmarks[i], original_filename)
                
                np_img = np.stack([np.array(batch['images'][i]).squeeze(0)] * 3, axis=-1) # reshape image to (384, 384, 3)
                #np_img = np.array(batch['images'][i])
                np_img = (((np_img + 1)/2)*255).astype(np.uint8) # Normalize to (0, 225)

                transformed = self.transform(image=np_img, keypoints=landmarks[i])
                transformed_image = transformed['image']
                transformed_keypoints = transformed['keypoints']
                

                # Reshape transformed_image from (384, 384, 3) to (1, 384, 384) 
                transformed_image = np.asarray(transformed_image)[:, :, 0]                
                # Reshape to (1, 384, 384)
                transformed_image = transformed_image[np.newaxis, :, :]

                #Update transformed_batch
                batch['images'][i] = transformed_image
                batch['seg_all'][i] = landmark_img_from_keypoints_with_boundaries(transformed_keypoints, 384, 16)/255

                #transformed_filename = os.path.join(output_dir, f"transformed_image_{i}.png")
                #save_image_with_landmarks(batch['images'][i].squeeze(0), transformed_keypoints, transformed_filename)
                
                #unique_values = torch.unique(batch['seg_all'][i])
                #print(f"Unique values in transformed seg_all[{i}]: {unique_values}")

                #plt.imshow(18*batch['seg_all'][i][0], cmap="gray")
                #plt.savefig("landmark visualization")


                #pdb.set_trace()

        
        # When returning, np_img should be shape (1, 384, 384) and 
        '''
        #pdb.set_trace()
        return {
            'images':batch['images'],
            'seg_all':batch['seg_all'], 
            'image_filenames':batch['image_filenames']
        }