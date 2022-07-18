"""
test the preprocessing methods in NiftiDataset

"""

from NiftiDataset import *
import NiftiDataset as NiftiDataset
from init import InitParser
import os
import matplotlib.pyplot as plt
import skimage.io as io
from ipywidgets import interact, interactive, IntSlider, ToggleButtons
import tqdm
from tqdm import *

file_path = "./data/raw_dataset2/train/ct/volume_1.nii.gz"


def read(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data


def show_img(image):
    len = image.shape[-1]
    plt.ion()
    for i in trange(len):
        plt.imshow(image[:, :, i], cmap='gray')
        plt.pause(0.0005)
    plt.ioff()
    plt.show()


def read_image(path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    image = reader.Execute()
    return image


def Normalize(file_path):
    bit = sitk.sitkFloat32
    Segmentation = False

    data_path = file_path
    label_path = file_path.replace('volume', 'segmentation').replace('ct', 'label')

    # read image and label
    image = read_image(data_path)

    image = Normalization(image)  # set intensity 0-255

    # cast image and label
    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(bit)
    image = castImageFilter.Execute(image)

    label = read_image(label_path)
    if Segmentation is False:
        label = Normalization(label)  # set intensity 0-255
    castImageFilter.SetOutputPixelType(bit)
    label = castImageFilter.Execute(label)

    sample = {'image': image, 'label': label}

    # if self.transforms:  # apply the transforms to image and label (normalization, resampling, patches)
    #     for transform in self.transforms:
    #         sample = transform(sample)

    # convert sample to tf tensors
    image_np = sitk.GetArrayFromImage(sample['image'])
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


def Normalize_resize(file_path):
    args = InitParser()
    bit = sitk.sitkFloat32
    Segmentation = True
    transforms = [
        # NiftiDataset.Resample(args.new_resolution, args.resample),
        # NiftiDataset.Resize(args.new_size, args.resize),
        NiftiDataset.Augmentation(),
        # NiftiDataset.Padding((args.patch_size[0], args.patch_size[1], args.patch_size[2])),
        # NiftiDataset.RandomCrop((args.patch_size[0], args.patch_size[1], args.patch_size[2]), args.drop_ratio,
        #                         args.min_pixel),
    ]

    data_path = file_path
    label_path = file_path.replace('volume', 'segmentation').replace('ct', 'label')

    # read image and label
    image = read_image(data_path)

    image = Normalization(image)  # set intensity 0-255

    # cast image and label
    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(bit)
    image = castImageFilter.Execute(image)

    label = read_image(label_path)
    # if Segmentation is False:
    label = Normalization(label)  # set intensity 0-255
    castImageFilter.SetOutputPixelType(bit)
    label = castImageFilter.Execute(label)

    sample = {'image': image, 'label': label}

    if transforms:  # apply the transforms to image and label (normalization, resampling, patches)
        for transform in transforms:
            sample = transform(sample)

    # convert sample to tf tensors
    image_np = sitk.GetArrayFromImage(sample['image'])
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


def test_Normalize():
    file_path = "./data/raw_dataset2/train/ct/volume_1.nii.gz"

    image2, label = Normalize(file_path)
    image2 = torch.squeeze(image2).numpy()
    label = torch.squeeze(label).numpy()
    show_img(image2)
    show_img(label)


def test_resize():
    file_path = "./data/raw_dataset2/train/ct/volume_1.nii.gz"

    image2, label = Normalize_resize(file_path)
    image2 = torch.squeeze(image2).numpy()
    label = torch.squeeze(label).numpy()
    show_img(image2)
    show_img(label)


if __name__ == '__main__':
    test_resize()
