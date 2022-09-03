import SimpleITK as sitk

path = '/media/wu24/Dy/DY/z_3dunet/Segmentation_Project/data/raw_dataset2/test/ct/volume_6.nii.gz'

image = sitk.ReadImage(path)
inputsize = image.GetSize()
inputspacing = image.GetSpacing()

print(inputsize)
print(inputspacing)
