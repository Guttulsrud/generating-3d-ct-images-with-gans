import glob
import os

import nibabel
import pandas as pd
from tqdm import tqdm

from evaluation.metrics import evaluate

note = 'NIR_real_CCA_fake'
experiment = '256_normalized15mm'
real_data = '../data/256/normalized_interpolated_resized/masks'
fake_data = f'../data/post_processed/{experiment}/cca_concat'

fid_score, is_score = evaluate(real_images_path=real_data,
                               generated_images_path=fake_data)

df = pd.DataFrame([[experiment, fid_score, is_score]], columns=['Experiment', 'FID', 'IS'])
filename = f'results/{experiment}_{note}.csv'

df.to_csv(filename, index=False)
