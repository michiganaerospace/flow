from scipy import signal
from scipy import misc
from glob import glob
from utils import load_image_file
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from statistics import mean, stdev

FOLDER = "waves40fps"
N = 100  # Frames to include in computation.
dN = 8  # Step between frames.


def extract_tiles(image_sample, tile_size=100, step_size=100):
    """Extract uniformly sampled tiles from the image sample."""
    # TODO fix to enforce all same size
    h, w = image_sample.shape
    rows = np.arange(0, h, step_size)
    cols = np.arange(0, w, step_size)
    g = np.meshgrid(cols, rows)
    cols_rows = list(zip(*(x.flat for x in g)))
    tiles = []
    for p in cols_rows:
        tiles.append(
            image_sample[
                p[1]: p[1] + tile_size,
                p[0]: p[0] + tile_size,
            ]
        )
    return tiles


def get_sequences():
    """Return an array of tile sequences across frames."""
    files = glob(f"./{FOLDER}/*")
    img_tiles = []
    for cnt in range(N):
        img = load_image_file(files[cnt])
        tiles = extract_tiles(img)
        img_tiles.append(tiles)
    sequences = []
    for indx in range(len(img_tiles[0])):
        sqc = [tile[indx] for tile in img_tiles]
        sequences.append(sqc)
    return sequences


if __name__ == "__main__":

    sequences = get_sequences()

    # Get distributions for the correlations between frames for each sequence.
    distributions = []
    for sequence in sequences:
        vals = []
        for i in range(dN, len(sequence)):
            tile_a = sequence[i-dN]
            tile_b = sequence[i]
            # Leave out edge tiles that are different shape.
            if tile_a.shape[0] == tile_a.shape[1]:
                sim, diff = structural_similarity(tile_a, tile_b, full=True)
                vals.append(sim)
                # plt.imshow(diff)
                # plt.show()
        if vals != []:
            distributions.append(vals)
        # plt.hist(vals)
        # plt.show()

    # Plot mean values for each distribution of similarity indices for a tile sequence.
    means = [mean(d) for d in distributions]
    plt.hist(means)
    plt.show()
