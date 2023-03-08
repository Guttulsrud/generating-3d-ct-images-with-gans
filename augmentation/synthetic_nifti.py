import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

image = nib.load('../data/unzipped/training/labels/CHUM-001.nii.gz')
data = image.get_fdata()

# Get the data for slice 57
slice_data = data[:, :, 56]

# Create a yellow square
square_size = 50
square = np.ones((square_size, square_size)) * 255

# Add the yellow square to the center of the slice
x_start = int(slice_data.shape[0] / 2 - square_size / 2)
y_start = int(slice_data.shape[1] / 2 - square_size / 2)
slice_data[x_start:x_start + square_size, y_start:y_start + square_size] = square

# Save the modified slice as a new NIfTI image
data[:, :, 56] = slice_data
new_img = nib.Nifti1Image(data, image.affine, image.header)
nib.save(new_img, 'modified_image.nii.gz')

# Display the modified slice
plt.imshow(slice_data, cmap='gray')
plt.show()

exit()
# # Create a 3D array with shape (10, 10, 10)
# data = np.zeros((512, 512, 91))
#
# # Generate circles in each slice of the third dimension
# for z in range(data.shape[2]):
#     x, y = np.meshgrid(np.linspace(-1, 1, data.shape[0]), np.linspace(-1, 1, data.shape[1]))
#     r = np.sqrt(x**2 + y**2)
#     circle = np.zeros_like(r)
#     circle[r <= 0.7] = 1
#     data[:, :, z] = circle
#
# # Save the 3D array as a NIfTI image
# img = nib.Nifti1Image(data, np.eye(4))
# nib.save(img, 'circle.nii.gz')
# print(img.shape)
# # Display one slice of the NIfTI image
# plt.imshow(data[:, :, 5])
# plt.show()
# exit()
# import nibabel as nib
# import numpy as np
#
# from utils.preprocessing_3d.preprocess_nifti_images import display_image

# Load the first NIfTI image
img1 = nib.load('square.nii.gz')
data1 = img1.get_fdata()

# Load the second NIfTI image
img2 = nib.load('circle.nii.gz')
data2 = img2.get_fdata()


# Concatenate the two arrays along axis 2
data_concat = np.concatenate((data1, data2), axis=2)

# Create a new NIfTI image from the concatenated data
affine = img1.affine
hdr = img1.header
img_concat = nib.Nifti1Image(data_concat, affine, hdr)

display_image(img_concat.get_fdata())

# Save the concatenated NIfTI image
nib.save(img_concat, 'concatenated.nii.gz')
img2 = nib.load('concatenated.nii.gz')
data2 = img2.get_fdata()
display_image(data2)
