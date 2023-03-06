import glob
import nibabel as nib
import os
from tqdm import tqdm

path = f'../../data'

image_paths = glob.glob(os.path.join(f'{path}/3d/preprocessed/images', '*CT.nii.gz'))
mask_paths = glob.glob(os.path.join(f'{path}/3d/preprocessed/masks', '*.nii.gz'))

for image_path, mask_path in tqdm(zip(image_paths, mask_paths)):
    path = os.path.join("../../data/3d/preprocessed/concatenated", image_path.split('images\\')[-1])
    path = path.replace('__CT', '')
    img1 = nib.load(image_path)
    img2 = nib.load(mask_path)

    if img1.shape != img2.shape:
        print(image_path, mask_path)
        continue

    try:
        concat_img = nib.concat_images([img1, img2], axis=2)
        nib.save(concat_img, path)
    except Exception:
        print(image_path)

# 196it [21:33,  6.30s/it]../../data/3d/preprocessed/images\CHUV-003__CT.nii.gz
# 198it [21:41,  5.48s/it]../../data/3d/preprocessed/images\CHUV-005__CT.nii.gz
# 199it [21:42,  4.23s/it]../../data/3d/preprocessed/images\CHUV-006__CT.nii.gz
# 204it [22:05,  4.35s/it]../../data/3d/preprocessed/images\CHUV-010__CT.nii.gz
# ../../data/3d/preprocessed/images\CHUV-011__CT.nii.gz
# 219it [23:34,  6.73s/it]../../data/3d/preprocessed/images\CHUV-026__CT.nii.gz
# 220it [23:35,  5.16s/it]../../data/3d/preprocessed/images\CHUV-027__CT.nii.gz
# 224it [23:58,  5.99s/it]../../data/3d/preprocessed/images\CHUV-031__CT.nii.gz
# 225it [23:59,  4.60s/it]../../data/3d/preprocessed/images\CHUV-032__CT.nii.gz
# 227it [24:06,  4.33s/it]../../data/3d/preprocessed/images\CHUV-034__CT.nii.gz
# 232it [24:29,  4.21s/it]../../data/3d/preprocessed/images\CHUV-038__CT.nii.gz
# 233it [24:35,  4.63s/it]../../data/3d/preprocessed/images\CHUV-040__CT.nii.gz
# 234it [24:36,  3.64s/it]../../data/3d/preprocessed/images\CHUV-041__CT.nii.gz
# 236it [24:43,  3.78s/it]../../data/3d/preprocessed/images\CHUV-043__CT.nii.gz
# 238it [24:46,  2.53s/it]../../data/3d/preprocessed/images\CHUV-044__CT.nii.gz
# ../../data/3d/preprocessed/images\CHUV-045__CT.nii.gz
# 239it [24:47,  2.18s/it]../../data/3d/preprocessed/images\CHUV-046__CT.nii.gz
# 242it [25:03,  4.83s/it]../../data/3d/preprocessed/images\CHUV-049__CT.nii.gz
# 244it [25:12,  4.81s/it]../../data/3d/preprocessed/images\CHUV-051__CT.nii.gz
# 249it [25:39,  5.63s/it]../../data/3d/preprocessed/images\HGJ-010__CT.nii.gz
# 264it [26:59,  5.59s/it]../../data/3d/preprocessed/images\HGJ-036__CT.nii.gz
# 266it [27:07,  4.81s/it]../../data/3d/preprocessed/images\HGJ-038__CT.nii.gz
# 267it [27:08,  3.72s/it]../../data/3d/preprocessed/images\HGJ-039__CT.nii.gz
# 276it [27:53,  5.32s/it]../../data/3d/preprocessed/images\HGJ-058__CT.nii.gz
# 281it [28:17,  5.17s/it]../../data/3d/preprocessed/images\HGJ-069__CT.nii.gz
# 290it [29:02,  5.45s/it]../../data/3d/preprocessed/images\HGJ-080__CT.nii.gz
# 291it [29:03,  4.17s/it]../../data/3d/preprocessed/images\HGJ-081__CT.nii.gz
# 295it [29:21,  4.79s/it]../../data/3d/preprocessed/images\HGJ-086__CT.nii.gz
# 296it [29:23,  3.70s/it]../../data/3d/preprocessed/images\HGJ-087__CT.nii.gz
# 320it [31:37,  5.72s/it]../../data/3d/preprocessed/images\MDA-001__CT.nii.gz
# 327it [32:17,  5.14s/it]../../data/3d/preprocessed/images\MDA-010__CT.nii.gz
# ../../data/3d/preprocessed/images\MDA-011__CT.nii.gz
# 347it [34:10,  6.58s/it]../../data/3d/preprocessed/images\MDA-031__CT.nii.gz ../../data/3d/preprocessed/masks\MDA-031.nii.gz
# 422it [41:52,  5.02s/it]../../data/3d/preprocessed/images\MDA-105__CT.nii.gz
# 435it [43:12,  6.15s/it]../../data/3d/preprocessed/images\MDA-120__CT.nii.gz
# 437it [43:19,  5.01s/it]../../data/3d/preprocessed/images\MDA-122__CT.nii.gz
# 439it [43:28,  4.92s/it]../../data/3d/preprocessed/images\MDA-124__CT.nii.gz
# 481it [47:36,  6.14s/it]../../data/3d/preprocessed/images\MDA-167__CT.nii.gz
# 483it [47:43,  5.08s/it]../../data/3d/preprocessed/images\MDA-169__CT.nii.gz
# 504it [49:46,  6.28s/it]../../data/3d/preprocessed/images\MDA-190__CT.nii.gz