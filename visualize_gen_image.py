import os

from utils.model.generate_image import generate_image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# from utils.model.generate_image import generate_image
from visualization.display_image import display_image

model_path = 'saved_models/concat/checkpoint'

for x in range(0, 10):
    display_image(generate_image(model_path=model_path))