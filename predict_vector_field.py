"""predict_vector_field.py
--
Process video frames to estimate velocity vector field.
"""
from vessel import Vessel

import cv2
import numpy as np
from skimage import feature
from tqdm import tqdm


# Point this to a folder containing the raw video frames (assumed jpeg).
LOCATION_OF_RAW_IMAGES = "./waves2997fps"
FRAME_RATE = 29.97  # FPS of original video
FRAME_INTERVAL = 5  # Compare images this many steps apart.
DELTA_TIME = FRAME_INTERVAL / FRAME_RATE


def load_image(image_number):
    path_to_image = f"{LOCATION_OF_RAW_IMAGES}/img{image_number}.jpeg"
    img = cv2.imread(path_to_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def estimate_velocity(first_image, second_image, dt):
    """Find the relative shift between the two images."""
    mask = np.ones_like(first_image).astype(bool)
    delta = feature.masked_register_translation(first_image, second_image, mask)
    return delta / dt


def extract_tiles(image, tile_size, step_size):
    """Extract uniformly sampled tiles from the image."""
    h, w = image.shape
    rows = np.arange(tile_size, h, step_size)
    cols = np.arange(tile_size, w, step_size)
    g = np.meshgrid(cols, rows)
    cols_rows = list(zip(*(x.flat for x in g)))
    tiles = []
    for p in cols_rows:
        tiles.append(
            image[
                p[1] - tile_size : p[1] + tile_size,
                p[0] - tile_size : p[0] + tile_size,
            ]
        )
    return tiles, cols_rows


def estimate_velocity_field(first_image, second_image, tile_size, step_size, dt):
    """Estimate velocity field across the image."""
    tiles_1, rows_cols = extract_tiles(first_image, tile_size, step_size)
    tiles_2, rows_cols = extract_tiles(second_image, tile_size, step_size)
    v_field = [estimate_velocity(t1, t2, dt) for t1, t2 in zip(tiles_1, tiles_2)]
    return rows_cols, v_field


def process_images(tile_size=100, max_number_images=1000, dt=DELTA_TIME):
    """Process multiple images."""
    crs = []
    vfs = []
    image_numbers = []
    for image_number in tqdm(np.arange(1, max_number_images)):
        img1 = load_image(image_number)
        img2 = load_image(image_number + FRAME_INTERVAL)
        cr, vf = estimate_velocity_field(img1, img2, tile_size, 200, dt)
        crs.append(cr)
        vfs.append(vf)
        image_numbers.append(image_number)

    # Save this data.
    crs = np.array(crs)
    vfs = np.array(vfs)
    v = Vessel("fields.dat")
    v.crs = crs
    v.vfs = vfs
    v.image_numbers = image_numbers
    v.save()


if __name__ == "__main__":

    # Run the process_images function to predict vector field on image sequences...
    process_images(max_number_images=3)
