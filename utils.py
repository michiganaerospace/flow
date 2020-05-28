"""utils.py
-- 
Image functions to use across files. 

"""

import cv2
import glob
import shutil
import os
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageEnhance


def load_image_file(path):
    """ Load img pixel info in matrix form."""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def load_image_num(image_number, folder):
    """ Load image specified by image_number.
        Note: Images should be saved in a folder with the following command: ffmpeg -i video_file_name -vf fps=video_fps img%d.jpeg. 
    """
    path_to_image = f"{folder}/img{image_number}.jpeg"
    return load_image_file(path_to_image)


def enhance(folder_name):
    """Enhance image pixel color values to give image more contrast and intensity."""
    waves = glob.glob(f"./{folder_name}/*.jpeg")
    os.mkdir(f"enhanced_{folder_name}")
    print("Enhancing images.")
    for file in tqdm(waves):
        im = Image.open(file)
        enhancer = ImageEnhance.Contrast(im)
        enhanced_im = enhancer.enhance(4.0)
        filename = file.replace(f"./{folder_name}/", "enhanced_")
        enhanced_im.save(filename)
        shutil.move(filename, f"./enhanced_{folder_name}")


def normalize(folder_name):
    """Normalize image pixel intensity."""
    waves = glob.glob(f"./{folder_name}/*.jpeg")
    os.mkdir(f"normalized_{folder_name}")
    print("Normalizing images.")
    for file in tqdm(waves):
        img = cv2.imread(file, 0)
        equ = cv2.equalizeHist(img)
        filename = file.replace(f"./{folder_name}/", "normalized_")
        cv2.imwrite(filename, equ)
        shutil.move(filename, f"./normalized_{folder_name}")
