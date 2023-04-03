import glob
import json
import random


def create_file_mapping(experiment, path):
    img_input_dir = f'{path}/images/'
    mask_input_dir = f'{path}/masks/'
    img_list = list(glob.glob(img_input_dir + "*.gz"))
    mask_list = list(glob.glob(mask_input_dir + "*.gz"))
    img_mask_pairs = list(zip(img_list, mask_list))

    random.shuffle(img_mask_pairs)

    size = len(img_mask_pairs) // 5

    # Split the list into sublists
    folds = [img_mask_pairs[i:i + size] for i in range(0, len(img_mask_pairs), size)]

    output = {
        'training': [],
        'testing': []
    }

    # Assign a fold number to each pair and append it to the 'training' key in output
    for i, fold in enumerate(folds):
        for pair in fold:
            im = pair[0].split("\\")[-1]
            im2 = pair[1].split("\\")[-1]

            output['training'].append({
                "image": f'images/{im}',
                "label": f'masks/{im2}',
                "fold": i
            })

    with open(f'{experiment}_folds.json', 'w') as f:
        json.dump(output, f)
