import glob
import os
# from medpy.metric.binary import dc, hd

import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import nibabel as nib

from visualization.display_image import display_image

gt_masks = glob.glob('../../data/128/ground_truth/masks/*.nii.gz')
pred_masks = glob.glob('data/128/thresholded_5_fold/*/*.nii.gz')

pred_names = []
for x in pred_masks:
    pred_names.append(x.split('_ensemble')[0].split('\\')[-1])

gt = []
for x in gt_masks:
    if x.split('\\')[-1].split('_')[0].replace('.nii.gz', '') in pred_names:
        gt.append(x)

gt_masks = gt

dsc_sum = 0
num_voxels_sum = 0
zero_masks = 0

dices = []
dices2 = []
dices3 = []

# Loop over the images and calculate the DSC for each
for gt_mask, pred_mask in tqdm(zip(gt_masks, pred_masks)):
    m_path = gt_mask
    p_path = pred_mask
    gt_mask = sitk.ReadImage(gt_mask)
    pred_mask = sitk.ReadImage(pred_mask)

    sitk_pred_data = sitk.GetArrayFromImage(pred_mask)
    sitk_gt_data = sitk.GetArrayFromImage(gt_mask)

    gt_mask = sitk.BinaryThreshold(gt_mask, 1, 1, 1)
    pred_mask = sitk.BinaryThreshold(pred_mask, 1, 1, 1)

    # Compute the label statistics
    stats_filter = sitk.LabelStatisticsImageFilter()
    stats_filter.Execute(gt_mask, pred_mask)

    num_voxels = stats_filter.GetCount(1)
    num_voxels_sum += num_voxels
    if num_voxels > 0:
        dsc = 2 * stats_filter.GetSum(1) / (stats_filter.GetCount(1) + stats_filter.GetCount(2))
        dsc_sum += dsc * num_voxels
        dices.append(dsc)

    dice = 2 * np.sum(sitk_pred_data * sitk_gt_data) / (np.sum(sitk_pred_data) + np.sum(sitk_gt_data))
    dices2.append(dice)

    if dice > 0:
        dices3.append(dice)

print('Aggregated DSC over dataset: ', dsc_sum / num_voxels_sum)
print('Mean dice: ', np.mean(dices2))
