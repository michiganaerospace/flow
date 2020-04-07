"""predict_vector_field.py
--
Process video frames to estimate velocity vector field. Saves the data to a
Vessel .dat file that is by default called fields.dat. This file will be used
by generate_video_frames.py to generate example frames.
"""
from vessel import Vessel

import cv2
import numpy as np
from skimage import feature
from tqdm import tqdm
import matplotlib.pyplot as plt


# Point this to a folder containing the raw video frames (assumed jpeg).
LOCATION_OF_RAW_IMAGES = "./waves40fps"  # "./waves2997fps"
FRAME_RATE = 29.97  # FPS of original video
FRAME_INTERVAL = 5  # Compare imaes this many steps apart.
DELTA_TIME = FRAME_INTERVAL / FRAME_RATE


def load_image(image_number):
    path_to_image = f"{LOCATION_OF_RAW_IMAGES}/img{image_number}.jpeg"
    img = cv2.imread(path_to_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def estimate_velocity(first_image, second_image, dt):
    """Find the relative shift between the two images."""
    mask = np.ones_like(first_image).astype(bool)
    delta = feature.masked_register_translation(
        first_image, second_image, mask)
    return delta / dt


def extract_tiles(image_sample, tile_size=100, step_size=100):
    """Extract uniformly sampled tiles 20x20 from the image sample 100x100."""
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
    return tiles, cols_rows


def kalman(v_field):
    """ Apply Kalman filtering on a v_field for an image tile."""
    n_iter = len(v_field)
    sz = (n_iter,)  # size of array
    Q = 1e-5  # process variance

    vx = [v[0] for v in v_field]
    vy = [v[1] for v in v_field]

    # allocate space for arrays
    xhat = np.zeros(sz)      # a posteri estimate of vx
    yhat = np.zeros(sz)      # and of vy
    Px = np.zeros(sz)         # a posteri error estimate of vx
    Py = np.zeros(sz)  # and of vy
    xhatminus = np.zeros(sz)  # a priori estimate of vx
    yhatminus = np.zeros(sz)  # and of vy
    Pxminus = np.zeros(sz)    # a priori error estimate vx
    Pyminus = np.zeros(sz)  # and of vy
    Kx = np.zeros(sz)         # gain or blending factor of vx
    Ky = np.zeros(sz)  # and of vy
    R = 0.1**2  # estimate of measurement variance, change to see effect

    # intial guesses
    xhat[0] = vx[0]
    yhat[0] = vy[0]
    Px[0] = 1.0
    Py[0] = 1.0

    for k in range(1, n_iter):
        # time update
        xhatminus[k] = xhat[k-1]
        yhatminus[k] = yhat[k-1]
        Pxminus[k] = Px[k-1]+Q
        Pyminus[k] = Py[k-1]+Q
        # measurement update
        Kx[k] = Pxminus[k]/(Pxminus[k]+R)
        Ky[k] = Pyminus[k]/(Pyminus[k] + R)
        xhat[k] = xhatminus[k]+Kx[k]*(vx[k]-xhatminus[k])
        yhat[k] = yhatminus[k] + Ky[k]*(vy[k] - yhatminus[k])
        Px[k] = (1-Kx[k])*Pxminus[k]
        Py[k] = (1-Ky[k])*Pyminus[k]

    # plt.figure()
    # plt.plot(vx, 'k+', label='noisy measurements')
    # plt.plot(xhat, 'b-', label='a posteri estimate')
    # plt.legend()
    # plt.title('vx estimate vs. iteration step', fontweight='bold')
    # plt.show()

    # plt.figure()
    # plt.plot(vy, 'k+', label='noisy measurements')
    # plt.plot(yhat, 'b-', label='a posteri estimate')
    # plt.legend()
    # plt.title('vy estimate vs. iteration step', fontweight='bold')
    # plt.show()

    return np.array([xhat[-1], yhat[-1]])


def estimate_velocity_field(first_image, second_image, tile_size, step_size, dt):
    """Estimate velocity field across the image."""
    img_field = []
    # For every tile of 100x100, sample 10x10 tiles.
    h, w = first_image.shape
    rows = np.arange(tile_size/2, h, step_size)
    cols = np.arange(tile_size/2, w, step_size)
    g = np.meshgrid(cols, rows)
    full_rc = list(zip(*(x.flat for x in g)))
    for r in np.arange(0, h, step_size):
        for j in np.arange(0, w, step_size):
            tile_first = first_image[r:r+tile_size, j:j+tile_size]
            tile_second = second_image[r:r+tile_size, j:j+tile_size]
            samples_first, rows_cols = extract_tiles(tile_first)
            samples_second, rows_cols = extract_tiles(tile_second)
            v_field = [estimate_velocity(t1, t2, dt)
                       for t1, t2 in zip(samples_first, samples_second)]
            img_field.append(kalman(v_field))
    return full_rc, img_field


def process_images(tile_size=300, step_size=100, max_number_images=10, dt=DELTA_TIME):
    """Process multiple images."""
    crs = []
    vfs = []
    image_numbers = []
    for image_number in tqdm(np.arange(1, max_number_images)):
        img1 = load_image(image_number)
        img2 = load_image(image_number + FRAME_INTERVAL)
        cr, vf = estimate_velocity_field(
            img1, img2, tile_size, step_size, dt)
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
    process_images()
