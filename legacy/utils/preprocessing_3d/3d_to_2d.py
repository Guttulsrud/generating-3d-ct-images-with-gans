import SimpleITK as sitk
import numpy as np
from PIL import Image
import glob
import os
from tqdm import tqdm


def image_3d_to_2d(path, mask=True):
    image = sitk.ReadImage(path)

    # Load the 3D ima
    # Get the size of the last dimension (depth)
    depth = image.GetSize()[2]

    # Create an empty numpy array to store the combined image
    combined_image = np.zeros((image.GetSize()[1] * depth, image.GetSize()[0]), dtype=np.uint16)

    # Loop through each slice and add it to the combined image
    for z in range(depth):
        # Get the slice as a numpy array
        slice_array = sitk.GetArrayFromImage(image[:, :, z])
        # Add the slice to the combined image
        combined_image[z * image.GetSize()[1]:(z + 1) * image.GetSize()[1], :] = slice_array

    combined_image = ((combined_image - np.min(combined_image)) / np.ptp(combined_image) * 255).astype(np.uint8)
    image_path = '2d/'
    image_path += 'images/' if not mask else 'masks/'
    image_path = image_path + path.split('\\')[1].split('.nii.gz')[0] + '.png'
    image_path = image_path.replace('CT', '')
    image_path = image_path.replace('__', '')

    pil_image = Image.fromarray(combined_image)
    # Save the PIL image as a PNG file
    pil_image.save(image_path)


images = glob.glob(os.path.join(f'../../../data/original/images', '*CT.nii.gz'))
labels = glob.glob(os.path.join(f'../../../data/original/masks', '*.nii.gz'))

for image_path, label_path in tqdm(zip(images, labels)):
    image_3d_to_2d(image_path, mask=False)
    image_3d_to_2d(label_path)
