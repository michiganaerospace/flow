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
    data = Vessel("fields.dat")
    image_location = "./videos"
    MAX_NUMBER_FRAMES = 10

    # Clear previous frames.
    imgs = glob("./videos/*.png")
    for img in imgs:
        os.remove(img)

    plt.ioff()
    for it in tqdm(np.arange(1, MAX_NUMBER_FRAMES)):
        plt.close("all")
        img = load_image_num(it, FOLDER_NAME)
        plt.imshow(img, cmap="bone")
        x = data.crs[it - 1, :, 1]
        y = data.crs[it - 1, :, 0]
        u = data.vfs[it - 1, :, 1]
        v = data.vfs[it - 1, :, 0]
        plt.quiver(y, x, u, v, color="yellow")
        plt.savefig(f"{image_location}/image_{it:04d}.png")
