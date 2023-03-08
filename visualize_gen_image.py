import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from utils.model.generate_image import generate_image
from visualization.display_image import display_image

model_path = 'saved_models/03-07 16-21_concat_128_test/checkpoint'

generated_image = generate_image(model_path=model_path)

display_image(generated_image)