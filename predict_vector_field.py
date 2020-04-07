"""predict_vector_field.py
--
Process video frames to estimate velocity vector field. Saves the data to a
Vessel .dat file that is by default called fields.dat. This file will be used
by generate_video_frames.py to generate example frames.

Takes argument FOLDER_NAME with raw wave images and FRAME_RATE.
"""

import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from vessel import Vessel
from skimage import feature
from tqdm import tqdm
from utils import enhance, load_image_num

FOLDER_NAME = sys.argv[1:][0]
FRAME_RATE = float(sys.argv[1:][1])
LOCATION_ENHANCED_IMGS = f"enhanced_{FOLDER_NAME}"
FRAME_INTERVAL = 5  # Compare images this many steps apart.
DELTA_TIME = FRAME_INTERVAL / FRAME_RATE


def estimate_velocity(first_image, second_image, dt):
    """Find the relative shift between the two images."""
    mask = np.ones_like(first_image).astype(bool)
    delta = feature.masked_register_translation(
        first_image, second_image, mask, overlap_ratio=3 / 10)
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
                p[1] - tile_size: p[1] + tile_size,
                p[0] - tile_size: p[0] + tile_size,
            ]
        )
    return tiles, cols_rows


def estimate_velocity_field(first_image, second_image, tile_size, step_size, dt):
    """Estimate velocity field across the image."""
    tiles_1, rows_cols = extract_tiles(first_image, tile_size, step_size)
    tiles_2, rows_cols = extract_tiles(second_image, tile_size, step_size)
    v_field = [estimate_velocity(t1, t2, dt)
               for t1, t2 in zip(tiles_1, tiles_2)]
    return rows_cols, v_field


def kalman(v_field):
    """ Apply Kalman filtering on a v_field for an image tile."""
    n_iter = len(v_field)
    sz = (n_iter,)  # size of array
    Q = 1e-5        # process variance

    vx = [v[0] for v in v_field]
    vy = [v[1] for v in v_field]

    # Allocate space for arrays.
    xhat = np.zeros(sz)       # a posteri estimate of vx
    yhat = np.zeros(sz)       # and of vy
    Px = np.zeros(sz)         # a posteri error estimate of vx
    Py = np.zeros(sz)         # and of vy
    xhatminus = np.zeros(sz)  # a priori estimate of vx
    yhatminus = np.zeros(sz)  # and of vy
    Pxminus = np.zeros(sz)    # a priori error estimate vx
    Pyminus = np.zeros(sz)    # and of vy
    Kx = np.zeros(sz)         # gain or blending factor of vx
    Ky = np.zeros(sz)         # and of vy
    R = 0.1**2                # estimate of measurement variance

    # Intial guesses.
    xhat[0] = vx[0]
    yhat[0] = vy[0]
    Px[0] = 1.0
    Py[0] = 1.0

    for k in range(1, n_iter):
        # Time update.
        xhatminus[k] = xhat[k-1]
        yhatminus[k] = yhat[k-1]
        Pxminus[k] = Px[k-1]+Q
        Pyminus[k] = Py[k-1]+Q
        # Measurement update.
        Kx[k] = Pxminus[k]/(Pxminus[k]+R)
        Ky[k] = Pyminus[k]/(Pyminus[k] + R)
        xhat[k] = xhatminus[k]+Kx[k]*(vx[k]-xhatminus[k])
        yhat[k] = yhatminus[k] + Ky[k]*(vy[k] - yhatminus[k])
        Px[k] = (1-Kx[k])*Pxminus[k]
        Py[k] = (1-Ky[k])*Pyminus[k]

    return np.matrix(tuple(zip(xhat, yhat)))


def process_images(tile_size=100, step_size=200, max_number_images=10, dt=DELTA_TIME):
    """Process multiple images."""
    crs = []
    vfs = []
    image_numbers = []
    print("Processing Images.")
    for image_number in tqdm(np.arange(1, max_number_images)):
        img1 = load_image_num(image_number)
        img2 = load_image_num(image_number + FRAME_INTERVAL)
        cr, vf = estimate_velocity_field(img1, img2, tile_size, step_size, dt)
        crs.append(cr)
        vfs.append(vf)
        image_numbers.append(image_number)

    # Kalman filter data for each tile.
    crs = np.array(crs)
    vfs = np.array(vfs)
    kalman_vfs = np.zeros_like(vfs)
    print("Applying Kalman filter.")
    for index in tqdm(np.arange(len(crs))):
        tile_field_series = vfs[:][index]
        kalman_vfs[:][index] = kalman(tile_field_series)

    # Save this data.
    v = Vessel("fields.dat")
    v.crs = crs
    v.vfs = kalman_vfs
    v.image_numbers = image_numbers
    v.save()


if __name__ == "__main__":

    # Work with enhanced images.
    if not os.path.isdir(LOCATION_ENHANCED_IMGS):
        enhance(FOLDER_NAME)

    # Run the process_images function to predict vector field on image sequences...
    process_images()
