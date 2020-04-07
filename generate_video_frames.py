"""generate_video_frames.py
--
Uses generated vector field data to generate video frames. By default these
frames are stored in /videos. To assemble these frames into a movie, navigate
to that directory and run:

$ ffmpeg -f image2 -r 30 -i image_%04d.png -vb 20M -vcodec mpeg4 -y movie_name.mp4
"""
import cv2
import numpy as np
import pylab as plt
from tqdm import tqdm

from argos.utils.vessel import Vessel


def load_image(image_number):
    path_to_image = f"waves40fps/img{image_number}.jpeg"
    img = cv2.imread(path_to_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


if __name__ == "__main__":

    # Load the previously calculated velocity field data.
    data = Vessel("fields.dat")
    image_location = "./videos"
    MAX_NUMBER_FRAMES = 10

    plt.ioff()
    for it in tqdm(np.arange(1, MAX_NUMBER_FRAMES)):
        plt.close("all")
        img = load_image(it)
        plt.imshow(img, cmap="bone")
        x = data.crs[it - 1, :, 1]
        y = data.crs[it - 1, :, 0]
        u = data.vfs[it - 1, :, 1]
        v = data.vfs[it - 1, :, 0]
        plt.quiver(y, x, u, v, color="yellow")
        plt.savefig(f"{image_location}/image_{it:04d}.png")
