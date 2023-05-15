import pandas as pd

from evaluation.metrics import evaluate

experiment = 'OG_images'
real_images = '128/normalized_interpolated_resized'
real_data = f'../data/{real_images}/images'

results = []

for x in ['OG_images', 'OG_images1x1x1', 'OG_normalized1.5x1.5x1.5', 'OG_normalized_2x2x2', 'OG_normalized_3x3x3',
          'OG_normalized_25x25x25']:
    fake_data = f'../data/generated_images/{x}/nifti'

    fid_score, is_score = evaluate(real_images_path=real_data,
                                   generated_images_path=fake_data, batch_size=32)
    results.append([x, fid_score, is_score])

    df = pd.DataFrame(results, columns=['Experiment', 'FID', 'IS'])
    df.to_csv(f'results/{x}.csv', index=False)
