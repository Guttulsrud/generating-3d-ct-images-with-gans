import os

from tqdm import tqdm

from utils.model.generate_image import generate_image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# from utils.model.generate_image import generate_image
from visualization.display_image import display_image

model_name = 'OG_normalized_25x25x25'
model_path = f'saved_models/{model_name}/checkpoint'

out_folder = f'generated_images/{model_name}'

if not os.path.exists('generated_images'):
    os.makedirs('generated_images')

if not os.path.exists(out_folder):
    os.makedirs(out_folder)

print('Generating images...')
for x in tqdm(range(0, 10)):
    gen_image = generate_image(model_path=model_path, save_step=80000)
    fig = display_image(gen_image, show=False, return_figure=True)
    fig.savefig(f'{out_folder}/{x + 1}.png')
