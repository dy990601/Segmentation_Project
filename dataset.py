import SimpleITK as sitk
import os
import re
import numpy as np
import random
import glob
import scipy.ndimage.interpolation as interpolation
from sklearn.model_selection import train_test_split
import scipy
import torch
import torch.utils.data
import torch.nn.functional as F
from skimage import measure
# import natsort as ns
import pandas as pd
from NiftiDataset import *

Segmentation = True


class MyNifitDataSet(torch.utils.data.Dataset):

    def __init__(self, data_list,
                 transforms=None,
                 train=False,
                 test=False, ):

        # Init membership variables
        self.data_list = data_list
        self.transforms = transforms
        self.train = train
        self.test = test
        self.bit = sitk.sitkFloat32

        """
        the dataset class receive a list that contain the data item, and each item
        is a dict with two item include data path and label path. as follow:
        data_list = [
        {
        "data":　data_path_1,
        "label": label_path_1,
        },
        {
        "data": data_path_2,
        "label": label_path_2,
        }
        ...
        ]
        """

    def read_image(self, path):
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        image = reader.Execute()
        return image

    def __getitem__(self, item):

        data_dict = self.data_list[item]
        data_path = data_dict["data"]
        label_path = data_dict["label"]

        # read image and label
        image = self.read_image(data_path)
        image = Normalization(image)  # set intensity 0-255

        # cast image and label
        castImageFilter = sitk.CastImageFilter()
        castImageFilter.SetOutputPixelType(self.bit)
        image = castImageFilter.Execute(image)

        if self.train:
            label = self.read_image(label_path)
            if Segmentation is False:
                label = Normalization(label)  # set intensity 0-255
            castImageFilter.SetOutputPixelType(self.bit)
            label = castImageFilter.Execute(label)

        elif self.test:
            label = self.read_image(label_path)
            if Segmentation is False:
                label = Normalization(label)  # set intensity 0-255
            castImageFilter.SetOutputPixelType(self.bit)
            label = castImageFilter.Execute(label)

        else:
            label = sitk.Image(image.GetSize(), self.bit)
            label.SetOrigin(image.GetOrigin())
            label.SetSpacing(image.GetSpacing())

        sample = {'image': image, 'label': label}

        if self.transforms:  # apply the transforms to image and label (normalization, resampling, patches)
            for transform in self.transforms:
                sample = transform(sample)

        # convert sample to tf tensors
        image_np = sitk.GetArrayFromImage(sample['image'])
        # image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        label_np = sitk.GetArrayFromImage(sample['label'])

        if Segmentation is True:
            label_np = abs(np.around(label_np))

        # to unify matrix dimension order between SimpleITK([x,y,z]) and numpy([z,y,x])  (actually it´s the contrary)
        image_np = np.transpose(image_np, (2, 1, 0))
        label_np = np.transpose(label_np, (2, 1, 0))

        image_np = image_np[np.newaxis, :, :, :]
        label_np = label_np[np.newaxis, :, :, :]

        # print(image_np.shape, label_np.shape)

        return torch.from_numpy(image_np), torch.from_numpy(label_np)  # this is the final output to feed the network

    def __len__(self):
        return len(self.data_list)