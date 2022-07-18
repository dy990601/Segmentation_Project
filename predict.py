from NiftiDataset import *
import NiftiDataset as NiftiDataset
from tqdm import tqdm
import datetime
from predict_single_image import from_numpy_to_itk, prepare_batch, inferenceFast
import math
from lifelines.utils import concordance_index


def read_image(path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    image = reader.Execute()
    return image


def inference_all(model, image_list, resize, size, segmentation):

    image = (image_list["data"])
    label = (image_list["label"])

    # a = (image.split('/')[-1]) # my pc
    # a = (a.split('\\')[-2])

    a = (image.split('/')[-2])  # dgx

    # if not os.path.isdir('./Data_folder/results'):
        # os.mkdir('./Data_folder/results')

    # print(image, a)
    # label_directory = os.path.join(str('./Data_folder/results/results_' + a + '.nii'))

    label_directory = image.replace(".nii", "_result.nii")

    result, dice = inferenceFast(False, model, image, label, './prova.nii', resize, size, segmentation=segmentation)

    # save segmented label
    writer = sitk.ImageFileWriter()
    writer.SetFileName(label_directory)
    writer.Execute(result)
    print("{}: Save evaluate label at {} success".format(datetime.datetime.now(), label_directory))
    print('************* Next image coming... *************')

    return dice


def inference_regression(model, image_list, isResize, size):

    model.eval()

    transforms = [
        NiftiDataset.Resize_regression(size, isResize)
    ]

    T1_image_path = image_list["T1"]
    T1c_image_path = image_list["T1c"]
    T2_image_path = image_list["T2"]
    PFS_label = image_list["PFS_LEVEL"]

    T1_image = read_image(T1_image_path)
    T1c_image = read_image(T1c_image_path)
    T2_image = read_image(T2_image_path)

    T1_image = Normalization(T1_image)
    T1c_image = Normalization(T1c_image)
    T2_image = Normalization(T2_image)

    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    T1_image = castImageFilter.Execute(T1_image)
    T1c_image = castImageFilter.Execute(T1c_image)
    T2_image = castImageFilter.Execute(T2_image)

    sample = {'T1': T1_image, 'T1c': T1c_image, "T2": T2_image}

    for transform in transforms:
        sample = transform(sample)

    T1_tfm, T1c_tfm, T2_tfm = sample['T1'], sample['T1c'], sample['T2']

    T1_np = sitk.GetArrayFromImage(T1_tfm)
    T1c_np = sitk.GetArrayFromImage(T1c_tfm)
    T2_np = sitk.GetArrayFromImage(T2_tfm)

    T1_np = np.transpose(T1_np, (2, 1, 0))
    T1c_np = np.transpose(T1c_np, (2, 1, 0))
    T2_np = np.transpose(T2_np, (2, 1, 0))

    T1_np = T1_np[np.newaxis, :, :, :]
    T1c_np = T1c_np[np.newaxis, :, :, :]
    T2_np = T2_np[np.newaxis, :, :, :]

    # print(T1_np.shape)
    input = torch.cat((torch.from_numpy(T1_np).cuda(),
                       torch.from_numpy(T1c_np).cuda(),
                       torch.from_numpy(T2_np).cuda()), dim=0).unsqueeze(0)

    output = model(input)
    PFS_predict = output.argmax(dim=1, keepdim=True).squeeze().data.cpu().numpy()

    return PFS_predict, PFS_label


def check_accuracy_model(model, images, resize, size):

    np_dice = []
    print("0/%i (0%%)" % len(images))
    for i in range(len(images)):

       Np_dice = inference_all(model=model, image_list=images[i], resize=resize, size=size, segmentation=True)

       np_dice.append(Np_dice)

    np_dice = np.array(np_dice)
    print('Mean volumetric DSC:', np_dice.mean())
    return np_dice.mean()

def check_accuracy_model_regression(model, images, resize, size):

    np_PFS_predict = []
    np_PFS_label = []
    for i in range(len(images)):

        Np_PFS_predict, Np_PFS_label  = inference_regression(model=model, image_list=images[i], isResize=resize, size=size)

        np_PFS_predict.append(Np_PFS_predict)
        np_PFS_label.append(Np_PFS_label)

    np_PFS_predict = np.array(np_PFS_predict)
    np_PFS_label = np.array(np_PFS_label)
    print(np_PFS_predict)
    print(np_PFS_label)
    C_index = concordance_index(np_PFS_label, np_PFS_predict)

    return C_index
