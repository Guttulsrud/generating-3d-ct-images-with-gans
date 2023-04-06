import nibabel as nib
from skimage import metrics

# Load the two datasets of 3D nifti images
dataset1 = [nib.load('image1.nii.gz'), nib.load('image2.nii.gz'), ...]
dataset2 = [nib.load('image3.nii.gz'), nib.load('image4.nii.gz'), ...]

# Compute the SSIM between each corresponding pair of images in the datasets
ssim_scores = []
for i in range(len(dataset1)):
    img1 = dataset1[i].get_fdata()
    img2 = dataset2[i].get_fdata()
    ssim = metrics.structural_similarity(img1, img2)
    ssim_scores.append(ssim)

# Compute the mean SSIM score across all image pairs
mean_ssim = sum(ssim_scores) / len(ssim_scores)

print('Mean SSIM:', mean_ssim)
