import os
import natsort as ns
import numpy as np
import SimpleITK as sitk
from skimage import measure
from scipy import ndimage
import threading
from threading import Lock, Thread


def read_image(path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    image = reader.Execute()
    return image


def save_image(image, path):
    writer = sitk.ImageFileWriter()
    writer.SetFileName(path)
    writer.Execute(image)
    print("save image: " + path)


def largestConnectComponent(bw_img):
    labeled_img = measure.label(bw_img, connectivity=2)
    props = measure.regionprops(labeled_img)

    numPix = []
    for ia in range(len(props)):
        numPix += [props[ia].area]

    # 像素最多的连通区域及其指引
    maxnum = max(numPix)
    index = numPix.index(maxnum)

    # 最大连通区域的bounding box
    return props[index].bbox


def dataTransform(image_path, seg_path):
    image = read_image(image_path)
    seg = read_image(seg_path)

    image_np = sitk.GetArrayFromImage(image)
    seg_np = sitk.GetArrayFromImage(seg)
    xMax, yMax, zMax = image_np.shape

    seg_np = ndimage.binary_opening(seg_np, structure=np.ones((3, 10, 10)))
    # seg_np = ndimage.binary_erosion(seg_np, structure=np.ones((3, 10, 10)))
    # seg_np = ndimage.binary_dilation(seg_np, structure=np.ones((4, 30, 30)))

    # print(seg_np.max(), seg_np.min(), seg_np.sum())
    try:
        x1, y1, z1, x2, y2, z2 = largestConnectComponent(seg_np)
        x1 = x1 - 2 if x1 - 2 > 0 else 0
        x2 = x2 + 2 if x2 + 2 < xMax else xMax
        y1 = y1 - 30 if y1 - 30 > 0 else 0
        y2 = y2 + 30 if y2 + 30 < yMax else yMax
        z1 = z1 - 30 if z1 - 30 > 0 else 0
        z2 = z2 + 30 if z2 + 30 < zMax else zMax
    except:
        x1, y1, z1 = int(xMax / 4), int(yMax / 4), int(zMax / 4)
        x2, y2, z2 = int(3 * xMax / 4), int(3 * yMax / 4), int(3 * zMax / 4)

    image_np_new = image_np[x1:x2, y1:y2, z1:z2]
    seg_np_new = seg_np[x1:x2, y1:y2, z1:z2].astype(np.float)

    image_new = sitk.GetImageFromArray(image_np_new)
    image_new.SetDirection(image.GetDirection())
    image_new.SetOrigin(image.GetOrigin())
    image_new.SetSpacing(image.GetSpacing())
    image_path_new = image_path.replace(".nii", "_loc.nii")
    save_image(image_new, image_path_new)

    seg_new = sitk.GetImageFromArray(seg_np_new)
    seg_new.SetDirection(seg.GetDirection())
    seg_new.SetOrigin(seg.GetOrigin())
    seg_new.SetSpacing(seg.GetSpacing())
    seg_path_new = seg_path.replace(".nii", "_loc.nii")
    save_image(seg_new, seg_path_new)


if __name__ == "__main__":
    data_dir = "./Data_folder/test_set"
    # data_list = os.listdir(data_dir)
    # data_list = ns.natsorted(data_list)
    data_list = ["1460", "1470"]

    for data in data_list:
        data_path = os.path.join(data_dir, data)

        # if os.listdir(data_path).__len__() is not 15:
        #     print(data)

        T1_path = os.path.join(data_path, "T1.nii")
        T1_Seg_path = os.path.join(data_path, "T1_result.nii")

        T2_path = os.path.join(data_path, "T2.nii")
        T2_Seg_path = os.path.join(data_path, "T2_result.nii")

        T1c_path = os.path.join(data_path, "T1c.nii")
        T1c_Seg_path = os.path.join(data_path, "T1c_result.nii")

        # t1 = threading.Thread(target=dataTransform, args=(T1_path, T1_Seg_path))
        # t2 = threading.Thread(target=dataTransform, args=(T1c_path, T1c_Seg_path))
        # t3 = threading.Thread(target=dataTransform, args=(T2_path, T2_Seg_path))
        # t1.start()
        # t2.start()
        # t3.start()
        # try:
        #     dataTransform(T1_path, T1_Seg_path)
        #     dataTransform(T1c_path, T1c_Seg_path)
        #     dataTransform(T2_path, T2_Seg_path)
        # except:
        #     print(data + "---------------------------------------------------------------------------------------------")

        dataTransform(T1_path, T1_Seg_path)
        dataTransform(T1c_path, T1c_Seg_path)
        dataTransform(T2_path, T2_Seg_path)
