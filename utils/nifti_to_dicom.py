import nibabel as nib
import pydicom
from pydicom.uid import generate_uid
from tqdm import tqdm

# image = nib.load(image_path).get_fdata()

import os
import pydicom
import nibabel as nib

image = 'CHUM-002__CT'
image_path = f'data/original/images/{image}.nii.gz'
nifti_img = nib.load(image_path)

# Extract data and metadata from NIfTI file
nifti_data = nifti_img.get_fdata()
nifti_affine = nifti_img.affine
nifti_header = nifti_img.header


# Set common DICOM metadata fields
common_fields = {
    'PatientName': 'Patient^Name',
    'PatientID': '12345',
    'Modality': 'CT',
    'BitsAllocated': 16,
    'BitsStored': 16,
    'HighBit': 15,
    'PixelRepresentation': 1,
    'SamplesPerPixel': 1,
    'PhotometricInterpretation': 'MONOCHROME2',
    'PixelSpacing': [nifti_header['pixdim'][1], nifti_header['pixdim'][2]],
    'SliceThickness': nifti_header['pixdim'][3],
}

# Loop over each volume in NIfTI image and create a DICOM Series
for i in range(nifti_data.shape[2]):
    # Create DICOM object
    dcm = pydicom.dataset.FileDataset(f'series/{i}.dcm', {}, file_meta=None, preamble=b'\0'*128)

    # Set DICOM metadata fields
    dcm.update(common_fields)
    dcm.SeriesInstanceUID = generate_uid()
    dcm.StudyInstanceUID = generate_uid()
    dcm.FrameOfReferenceUID = generate_uid()
    dcm.InstanceNumber = i + 1

    # Set DICOM ImageType field
    dcm.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
    if nifti_header['intent_code'] == 1007:
        dcm.ImageType.append('DIFFUSION')
        dcm.ImageType.append('FA')
    if 'TR' in nifti_header:
        dcm.ImageType.append('CARDIAC')
    if nifti_header['dim'][4] > 1:
        dcm.ImageType.append('VOLUME')

    # Set DICOM pixel data for current volume
    dcm.PixelData = nifti_data[..., i].astype('int16').tobytes()

    # Save DICOM object to file
    dcm.save_as(f'series/{i}.dcm')
