"""generate_video_frames.py
--
Uses generated vector field data to generate video frames. By default these
frames are stored in /videos. 

Takes argument FOLDER_NAME where raw wave images are stored.

To assemble these frames into a movie, navigate to that directory and run:

$ ffmpeg -f image2 -r 30 -i image_%04d.png -vb 20M -vcodec mpeg4 -y movie_name.mp4
"""
import cv2
import os
import sys
import numpy as np
import pylab as plt
from tqdm import tqdm
from glob import glob
from argos.utils.vessel import Vessel
from utils import load_image_num

FOLDER_NAME = sys.argv[1:][0]

if __name__ == "__main__":

    # Load the previously calculated velocity field data.
    data_kalman = Vessel("fields_kalman.dat")
    image_location_kalman = "./videos/kalman"
    data_weighted = Vessel("fields_weighted.dat")
    image_location_weighted = "./videos/weighted"
    data_update = Vessel("fields_update.dat")
    image_location_update = "./videos/updated"
    MAX_NUMBER_FRAMES = 100

    # Clear previous frames.
    imgs = glob("./videos/**/*.png")
    for img in imgs:
        os.remove(img)

    plt.ioff()
    for it in tqdm(np.arange(1, MAX_NUMBER_FRAMES)):
        img = load_image_num(it, FOLDER_NAME)
        plt.close("all")
        plt.imshow(img, cmap="bone")
        x = data_kalman.crs[it - 1, :, 1]
        y = data_kalman.crs[it - 1, :, 0]
        u = data_kalman.vfs[it - 1, :, 1]
        v = data_kalman.vfs[it - 1, :, 0]
        plt.quiver(y, x, u, v, color="yellow")
        plt.savefig(f"{image_location_kalman}/image_{it:04d}.png")

        plt.close("all")
        plt.imshow(img, cmap="bone")
        x = data_weighted.crs[it - 1, :, 1]
        y = data_weighted.crs[it - 1, :, 0]
        u = data_weighted.vfs[it - 1, :, 1]
        v = data_weighted.vfs[it - 1, :, 0]
        plt.quiver(y, x, u, v, color="yellow")
        plt.savefig(f"{image_location_weighted}/image_{it:04d}.png")

        plt.close("all")
        plt.imshow(img, cmap="bone")
        x = data_update.crs[it - 1, :, 1]
        y = data_update.crs[it - 1, :, 0]
        u = data_update.vfs[it - 1, :, 1]
        v = data_update.vfs[it - 1, :, 0]
        plt.quiver(y, x, u, v, color="yellow")
        plt.savefig(f"{image_location_update}/image_{it:04d}.png")
