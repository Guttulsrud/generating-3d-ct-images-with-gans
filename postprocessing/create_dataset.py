import json
import os
import random

real_images_path = '../data/cropped_original'
generated_images_path = '../data/post_processed/scaled'

real_images = os.path.join(real_images_path, "images/")
fake_images = os.path.join(generated_images_path, "images/")

datalist_json = {"testing": [], "training": []}

for file in os.listdir(fake_images):
    datalist_json["training"].append(
        {"image": "images/" + file, "label": "masks/" + file, "fold": 0})  # Initialize as single fold


random.seed(42)
random.shuffle(datalist_json["training"])

num_folds = 5
fold_size = len(datalist_json["training"]) // num_folds
for i in range(num_folds):
    for j in range(fold_size):
        datalist_json["training"][i * fold_size + j]["fold"] = i

with open("datalist.json", "w", encoding="utf-8") as f:
    json.dump(datalist_json, f, ensure_ascii=False, indent=4)