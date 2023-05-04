import time

import yaml

from augmentation.Augmentor import Augmentor

with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)['augmentation']


aug = Augmentor(config)
start_time_total = aug.init()
for index, image_mask_path in enumerate(aug.data):
    print(f'[{index + 1}/{len(aug.data)}]', end='')
    start_time = time.time()
    data = aug.load_image_mask(image_mask_path)

    if config['normalize']:
        normalized = aug.normalize(data, voxels=(1.5, 1.5, 1.5))
        aug.save_image_mask(normalized, '128_normalized15mm', scale_intensity=True)

    if config['smooth_gaussian']:
        smoothed = aug.smooth_gaussian(data)
        aug.save_image_mask(smoothed, 'norm_smooth_gaussian')

    if config['rotate90']:
        rotated = aug.rotate90(normalized)
        aug.save_image_mask(rotated, 'norm_rotate90')

    if config['flip']:
        flipped = aug.flip(normalized, spatial_axis=1)
        aug.save_image_mask(flipped, 'norm_flipped_y')

    if config['rotate']:
        rotated = aug.rotate(data, display_image=True)
        aug.save_image_mask(rotated, 'norm_rotated')

    if config['translate']:
        translated = aug.random_translation(data, translate_range=(40, 40, 2), display_image=True)
        aug.save_image_mask(translated, 'norm_translated')

    if config['elastic_deform']:
        elastic_deformed = aug.random_elastic_deformation(data)
        aug.save_image_mask(elastic_deformed, 'norm_elastic_deformed')

    if config['reorient']:
        reoriented = aug.reorient_axes(data)
        aug.save_image_mask(reoriented, 'norm_reoriented')

    if config['random_affine']:
        affine_transformation = aug.random_affine_transformation(normalized)
        aug.save_image_mask(affine_transformation, 'norm_affine_transformation')

    end_time = time.time()
    runtime_seconds = end_time - start_time_total
    runtime_minutes, runtime_seconds = divmod(runtime_seconds, 60)
    runtime_hours, runtime_minutes = divmod(runtime_minutes, 60)
    print(
        f' [It: {int(end_time - start_time)}s]'
        f' [Total: {float(runtime_hours)}h,'
        f' {int(runtime_minutes)}m,'
        f' {int(runtime_seconds)}s]')
