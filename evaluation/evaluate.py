import os

import pandas as pd
from tqdm import tqdm

from evaluation.test import evaluate

filename = 'hpo_part2.csv'

results = []

folders = [f for f in os.listdir('../data/generated_images')]
folder_names = [f for f in folders]
for experiment in tqdm(folder_names):
    fid_score, is_score = evaluate(real_images_path='../data/concat',
                                   generated_images_path=f'../data/generated_images/{experiment}/nifti')

    results.append([experiment, fid_score, is_score])

df = pd.DataFrame(results, columns=['Experiment', 'FID', 'IS'])
df.to_csv(filename, index=False)
