from scipy import ndimage
import numpy as np


def connected_component_analysis(image, threshold, size_threshold):
    binary_mask = image > threshold

    # Label the connected components in the binary mask
    labeled_mask, num_labels = ndimage.label(binary_mask)

    # Compute the size of each connected component
    component_sizes = ndimage.sum(binary_mask, labeled_mask, range(1, num_labels + 1))

    # Keep only the connected components that meet a certain size threshold
    filtered_mask = np.zeros_like(labeled_mask)
    #filtered_mask = np.full_like(labeled_mask, fill_value=-1)

    for i in range(1, num_labels + 1):
        if component_sizes[i - 1] >= size_threshold:
            filtered_mask[labeled_mask == i] = 1

    # Apply the filtered mask to the original image to remove noise
    return image * filtered_mask
