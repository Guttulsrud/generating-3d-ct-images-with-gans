import json
import os
import random

import yaml

real_images_path = '../data/128/ground_truth'
generated_images_path = '../data/generated_images/128_H17_binary_mask_non_interpolateBASELINE/nifti/post_processed'
with open('val_data.json', 'r') as f:
    val_data = yaml.safe_load(f)

real_images = os.path.join(real_images_path, "images/")
fake_images = os.path.join(generated_images_path, "images/")
datalist_json = {"testing": [], "training": []}

real_images = os.listdir(real_images)
fake_images = os.listdir(fake_images)

print(len(real_images))

# rl = []
#
# real_training_images = []
# for file in real_images:
#     if f'{file.split(".")[0]}__CT.nii' not in val_data:
#         real_training_images.append(file)
#         continue
#
#     rl.append(file)
#
# real_images = rl
random.shuffle(real_images)
random.shuffle(fake_images)
# random.shuffle(real_training_images)
fake_images = fake_images[:419]

# real_testing_images = real_images[:len(real_images) // 2]
# real_training_images = real_images[len(real_images) // 2:]
split_idx = int(0.8 * len(real_images))
real_training_images = real_images[:split_idx]
real_testing_images = real_images[split_idx:]

for file in real_training_images + fake_images:
    # for file in fake_images + real_training_images:
    datalist_json["training"].append(
        {"image": "images/" + file, "label": "masks/" + file, "fold": 0})  # Initialize as single fold

num_folds = 5
fold_size = len(datalist_json["training"]) // num_folds

random.shuffle(datalist_json["training"])
for i in range(num_folds):
    for j in range(fold_size):
        datalist_json["training"][i * fold_size + j]["fold"] = i

for file in real_testing_images:
    datalist_json["testing"].append(
        {"image": "images/" + file, "label": "masks/" + file})

print(len(datalist_json["testing"]))
print(len(datalist_json["training"]))

with open("train_rf_test_r.json", "w", encoding="utf-8") as f:
    json.dump(datalist_json, f, ensure_ascii=False, indent=4)

# with open("datalist_test_full_dataset.json", "r", encoding="utf-8") as f:
#     datalist_json = json.load(f)
