import pandas as pd
from evaluation.metrics import evaluate
experiment = '128_H17_binary_mask_non_interpolateBASELINE'
real_images = '128/interpolated_resized'

print('Experiment:', experiment)

results = []
for mode in ['images', 'masks']:
    real_data = f'../data/{real_images}/{mode}'
    fake_data = f'../data/generated_images/{experiment}/nifti/post_processed/{mode}'

    fid_score, is_score = evaluate(real_images_path=real_data,
                                   generated_images_path=fake_data, batch_size=32)
    print(mode, fid_score, is_score)
    results.append([mode, fid_score, is_score])
df = pd.DataFrame(results, columns=['Experiment', 'FID', 'IS'])
df.to_csv(f'results/{experiment}.csv', index=False)
