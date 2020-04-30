from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm
import torch
from PIL import Image
import random
import cv2
import math
import pdb


class dataset(Dataset):

    """Face Landmarks dataset."""

    def __init__(self, datalist_file, root_dir, transform=None, with_path=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.with_path = with_path
        self.datalist_file =  datalist_file
        self.image_list, self.label_list = \
            self.read_labeled_image_list(self.root_dir, self.datalist_file)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform is not None:
            # image = self.transform(np.array(image))
            image = self.transform(image)

        if self.with_path:
            return img_name, image, self.label_list[idx]
        else:
            return image, self.label_list[idx]

    def read_labeled_image_list(self, data_dir, data_list):
        """
        Reads txt file containing paths to images and ground truth masks.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

        Returns:
          Two lists with all file names for images and masks, respectively.
        """
        f = open(data_list, 'r')
        img_name_list = []
        img_labels = []
        for line in f:
            if ';' in line:
                image, labels = line.strip("\n").split(';')
            else:
                if len(line.strip().split()) == 2:
                    image, labels = line.strip().split()
                    if '.' not in image:
                        image += '.JPEG'
                        # image += '.jpg'
                    labels = int(labels)
                else:
                    line = line.strip().split()
                    image = line[0] + ".JPEG"
                    labels = list(map(int, line[1:]))
            # img_name_list.append(os.path.join(data_dir, image))
            img_name_list.append(image)
            img_labels.append(labels)
        return img_name_list, np.array(img_labels, dtype=np.float32)

def get_name_id(name_path):
    name_id = name_path.strip().split('/')[-1]
    name_id = name_id.strip().split('.')[0]
    return name_id


