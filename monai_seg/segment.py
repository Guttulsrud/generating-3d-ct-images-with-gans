import json
import os

import yaml

from monai_seg.utils.create_json import create_file_mapping

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# from monai.apps.auto3dseg import AutoRunner

experiment_name = 'testing'
image_source = '../data/original'
if __name__ == "__main__":
    with open('source/input.yaml', 'r') as f:
        config_file = yaml.safe_load(f)

    create_file_mapping(experiment=experiment_name, path=image_source)
    config_file['dataroot'] = image_source
    config_file['datalist'] = f'{experiment_name}_folds.json'

    with open(f'{experiment_name}_input.yaml', 'w') as outfile:
        yaml.dump(config_file, outfile, default_flow_style=False)

    exit()
    runner = AutoRunner(input=f'{experiment_name}_input.yaml',
                        algos="segresnet",
                        work_dir=f"experiments/{experiment_name}",
                        ensemble=False)
    runner.run()
