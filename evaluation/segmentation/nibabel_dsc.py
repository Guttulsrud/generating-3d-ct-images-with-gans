import glob
import numpy as np
from tqdm import tqdm
import nibabel as nib
experiments = [
    'RF_128_fake_419_real',
    '524',
    '1024',
    '2048',
    'real_only5fold',
    'real_only_1fold',
    'RF',
    'RF_5_fold',
    'thresholded',
    'thresholded_5_fold'
]


gt_masks = glob.glob('../../data/128/ground_truth/masks/*.nii.gz')

for experiment in experiments:
    pred_masks = glob.glob(f'data/128/{experiment}/*/*.nii.gz')
    pred_names = []
    for x in pred_masks:
        pred_names.append(x.split('_ensemble')[0].split('\\')[-1])

    gt = []
    for x in gt_masks:
        if x.split('\\')[-1].split('_')[0].replace('.nii.gz', '') in pred_names:
            gt.append(x)

    dsc_sum = 0
    num_voxels_sum = 0
    zero_masks = 0


    dices = []
    dices2 = []
    dices3 = []

    for gt_path, pred_path in tqdm(zip(gt, pred_masks)):
        gt_mask = nib.load(gt_path).get_fdata().astype(bool)
        pred_mask = nib.load(pred_path).get_fdata().astype(bool)

        intersection = np.logical_and(gt_mask, pred_mask)
        union = np.logical_or(gt_mask, pred_mask)

        num_voxels = np.sum(gt_mask)
        num_voxels_sum += num_voxels

        dsc = 2 * np.sum(intersection) / np.sum(union)

        dsc_sum += dsc * num_voxels
        dices.append(dsc)

        dice = 2 * np.sum(gt_mask * pred_mask) / (np.sum(gt_mask) + np.sum(pred_mask))
        dices2.append(dice)

        if dice > 0:
            dices3.append(dice)

    print(dsc_sum / num_voxels_sum)



