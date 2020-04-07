import cv2
import glob
import shutil
import os
from tqdm import tqdm
from PIL import Image, ImageEnhance

FOLDER = "waves2997fps"


def load_image_file(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def load_image_num(image_number):
    path_to_image = f"{FOLDER}/img{image_number}.jpeg"
    return load_image_file(path_to_image)


def enhance(folder_name):
    waves = glob.glob(f"./{folder_name}/*.jpeg")
    os.mkdir(f"enhanced_{folder_name}")
    print("Enhancing images.")
    for file in tqdm(waves):
        im = Image.open(file)
        enhancer = ImageEnhance.Contrast(im)
        enhanced_im = enhancer.enhance(4.0)
        filename = file.replace("./folder_name/", "")
        enhanced_im.save(filename)
        shutil.move(filename, f"./enhanced_{folder_name}")
