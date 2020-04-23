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
from skimage.metrics import structural_similarity
from utils import enhance, load_image_num

FOLDER_NAME = sys.argv[1:][0]
FRAME_RATE = float(sys.argv[1:][1])
FRAME_INTERVAL = 5  # Compare images this many steps apart.
DELTA_TIME = FRAME_INTERVAL / FRAME_RATE
CORRELATION_THRESHOLD = 0.85


def estimate_velocity(first_image, second_image, dt):
    """Find the relative shift between the two tiles."""
    # Use structural similarity index as a weight for frame transaltion computation.
    sim, diff = structural_similarity(first_image, second_image, full=True)
    mask = np.ones_like(first_image).astype(bool)
    delta = feature.masked_register_translation(
        first_image, second_image, mask, overlap_ratio=3 / 10)
    return delta / dt, sim


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
    results = [estimate_velocity(t1, t2, dt)
               for t1, t2 in zip(tiles_1, tiles_2)]
    v_field = [r[0] for r in results]
    weights = [r[1] for r in results]
    return rows_cols, v_field, weights


def kalman(v_field):
    """ Apply Kalman filtering based on structural similarity on a v_field sequence for an image tile."""
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
        # Weight for raw values.
        # Measurement update.
        Kx[k] = Pxminus[k]/(Pxminus[k]+R)
        Ky[k] = Pyminus[k]/(Pyminus[k] + R)
        xhat[k] = xhatminus[k]+Kx[k]*(vx[k]-xhatminus[k])
        yhat[k] = yhatminus[k] + Ky[k]*(vy[k]-yhatminus[k])
        Px[k] = (1-Kx[k])*Pxminus[k]
        Py[k] = (1-Ky[k])*Pyminus[k]

    # Return filtered sequence with applied weights.
    return np.matrix(tuple(zip(xhat, yhat)))


def weighted_avg(v_field, weights):
    vx = [v[0] for v in v_field]
    vy = [v[1] for v in v_field]
    avg = []
    num_x = 0
    num_y = 0
    denom = 0
    for i in range(0, len(v_field)):
        num_x += weights[i]*vx[i]
        num_y += weights[i]*vy[i]
        denom += weights[i]
        avg.append([num_x/denom, num_y/denom])

    avg_x = [a[0] for a in avg]
    avg_y = [a[1] for a in avg]

    return np.matrix(tuple(zip(avg_x, avg_y)))


def update_vfs(v_field, weights):
    ''' Keep running weighted avg for previous 10 tiles. If computed value is outside accepted threshold, use computed avg.'''
    vx = [v[0] for v in v_field]
    vy = [v[1] for v in v_field]
    avg = [[vx[0], vy[0]]]
    num_x = 0
    num_y = 0
    denom = 0
    for i in range(1, len(v_field)):
        if len(avg) == 10:
            avg = [[vx[i-1], vy[1-1]]]

        # Compute percent difference.
        diff_x = abs(vx[i] - avg[i-1][0])/((vx[i]+avg[i-1][0])/2)
        diff_y = abs(vy[i] - avg[i-1][1])/((vy[i]+avg[i-1][1])/2)
        if diff_x > (1 - CORRELATION_THRESHOLD) or diff_y > (1 - CORRELATION_THRESHOLD):
            # Change calculation.
            v_field[i] = avg[i-1]

        # Update averages.
        num_x += weights[i]*vx[i]
        num_y += weights[i]*vy[i]
        denom += weights[i]
        avg.append([num_x/denom, num_y/denom])

    return v_field


def process_images(tile_size=100, step_size=100, max_number_images=10, dt=DELTA_TIME):
    """Process multiple images."""
    crs = []
    vfs = []
    weights = []
    image_numbers = []
    print("Processing Images.")
    for image_number in tqdm(np.arange(1, max_number_images)):
        img1 = load_image_num(image_number, FOLDER_NAME)
        img2 = load_image_num(image_number + FRAME_INTERVAL, FOLDER_NAME)
        cr, vf, sims = estimate_velocity_field(
            img1, img2, tile_size, step_size, dt)
        crs.append(cr)
        vfs.append(vf)
        weights.append(sims)
        image_numbers.append(image_number)

    # Kalman filter data for each tile.
    crs = np.array(crs)
    vfs = np.array(vfs)
    filtered_vfs = np.zeros_like(vfs)
    print("Applying filter.")
    n_tiles = len(crs[0])
    for index in tqdm(np.arange(n_tiles)):
        tile_field_series = [vf[index] for vf in vfs]
        tile_weight_series = [w[index] for w in weights]

        # weighted_avg(tile_field_series, tile_weight_series)
        # kalman(tile_field_series)
        new_vfs = update_vfs(tile_field_series, tile_weight_series)
        for seq in range(len(filtered_vfs)):
            filtered_vfs[seq][index] = new_vfs[seq]

    # Save this data.
    v = Vessel("fields.dat")
    v.crs = crs
    v.vfs = filtered_vfs
    v.image_numbers = image_numbers
    v.save()


if __name__ == "__main__":

    # Run the process_images function to predict vector field on image sequences...
    process_images()
