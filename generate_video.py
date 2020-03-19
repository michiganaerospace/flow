import cv2
import numpy as np
import pylab as plt
from tqdm import tqdm

from argos.utils.vessel import Vessel


def load_image(image_number):
    path_to_image = f"waves2997fps/img{image_number}.jpeg"
    img = cv2.imread(path_to_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


if __name__ == "__main__":

    data = Vessel("fields.dat")
    image_location = "./videos"

    plt.ioff()
    for it in tqdm(np.arange(1, 1000)):
        plt.close("all")
        img = load_image(it)
        plt.imshow(img, cmap="bone")
        x = data.crs[it - 1, :, 1]
        y = data.crs[it - 1, :, 0]
        u = data.vfs[it - 1, :, 1]
        v = data.vfs[it - 1, :, 0]
        plt.quiver(y, x, u, v, color="yellow")
        plt.savefig(f"{image_location}/image_{it:04d}.png")
