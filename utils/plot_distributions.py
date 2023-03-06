import glob
import os
from scipy.ndimage import zoom
import matplotlib as mpl

path = f'../data'
mpl.use('TkAgg')  # !IMPORTANT
import ast
import matplotlib.pyplot as plt

img_vals = []
label_vals = []
with open('e.txt', 'r') as f:
    # Read the file contents and split the rows by newline
    contents = f.read()
    rows = contents.split('\n')

    # Loop over the rows and evaluate each as a Python list
    for row in rows:
        if len(row) > 0:  # skip empty rows
            lst = ast.literal_eval(row)
            img_vals.append(lst)

print(img_vals)
plt.hist(img_vals)
plt.gca().set(title='Image dist', ylabel='Number of images')
plt.show()
exit()