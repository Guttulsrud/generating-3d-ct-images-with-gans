import os


def create_out_folders(experiment):
    out_folder = f'data/generated_images/{experiment}'

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    if not os.path.exists(f'../data/post_processed/{experiment}'):
        os.makedirs(f'../data/post_processed/{experiment}')

    if not os.path.exists(f'../data/post_processed/{experiment}/images'):
        os.makedirs(f'../data/post_processed/{experiment}/images')

    if not os.path.exists(f'../data/post_processed/{experiment}/masks'):
        os.makedirs(f'../data/post_processed/{experiment}/masks')

    if not os.path.exists(f'../data/post_processed/{experiment}/masks_cca'):
        os.makedirs(f'../data/post_processed/{experiment}/masks_cca')

    if not os.path.exists(f'../data/post_processed/{experiment}/masks_binary'):
        os.makedirs(f'../data/post_processed/{experiment}/masks_binary')

    if not os.path.exists(f'../data/post_processed/{experiment}/concat'):
        os.makedirs(f'../data/post_processed/{experiment}/concat')

    if not os.path.exists(f'../data/post_processed/{experiment}/concat_binary'):
        os.makedirs(f'../data/post_processed/{experiment}/concat_binary')

    if not os.path.exists(f'../data/post_processed/{experiment}/concat_cca'):
        os.makedirs(f'../data/post_processed/{experiment}/concat_cca')

    return out_folder
