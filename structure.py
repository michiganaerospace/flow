"""structure.py
--
Visualizes change in intensity for specified locations over a sequence of images. 

akes argument FOLDER_NAME where raw wave images are stored.
"""

from utils import load_image_num
from tqdm import tqdm
import sys
import numpy as np
import matplotlib.pyplot as plt

FOLDER_NAME = sys.argv[1:][0]
# Specify step size between images grabbed from the sequence. Change to inspect different plots.
FRAME_INTERVAL = 1


def get_tile_avg(img, loc):
    """For a tile of length 4 centered at loc on img, get an average of pixel intensity."""
    size = 2
    vals = []
    # Collect values within tile.
    for r in np.arange(loc[0] - size, loc[0] + size):
        for c in np.arange(loc[1] - size, loc[1]+size):
            vals.append(img[r, c])
    # Return avg.
    return np.sum(vals)/len(vals)


def get_intensity_sequence(max_number_images=1800):
    """Return a sequence of intensity values for locations across a max_number_images length sequence of images."""
    # Get images.
    imgs = []
    for image_number in tqdm(np.arange(1, max_number_images, FRAME_INTERVAL)):
        imgs.append(load_image_num(image_number, FOLDER_NAME))
    # Set different pixel locations in reference to image.
    h, w = imgs[0].shape
    l1 = [int(h/2), int(w/2)]
    l2 = [int(h/4), int(w/4)]
    l3 = [h-int(h/4), w-int(w/4)]
    l4 = [h-int(h/4), int(w/4)]
    l5 = [int(h/4), w-int(w/4)]
    locs = [l1, l2, l3, l4, l5]
    # Get intensity sequence for each pixel location.
    for loc in locs:
        sequence = []
        for img in imgs:
            sequence.append(get_tile_avg(img, loc))
        # Idea is that it might look like a wave function, but in reality is very noisy.
        plt.plot(sequence)
        plt.show()


if __name__ == "__main__":
    get_intensity_sequence()
