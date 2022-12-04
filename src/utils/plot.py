import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

test_load = nib.load('file.nii').get_fdata()
for i in range(5):
    plt.subplot(5, 5, i + 1)
    plt.imshow(test_load[:, :, 55 + i])
    plt.gcf().set_size_inches(10, 10)
plt.show()
