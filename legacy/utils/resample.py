import nibabel as nib
from scipy import ndimage

image = nib.load('CHUM-001__CT.nii.gz')
percent = 0.03
image_data = image.get_fdata()

resampled_image = ndimage.zoom(image_data, (percent, percent, percent))
print(resampled_image.shape)
