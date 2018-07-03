import os

import cmd

from inception_score import get_inception_score

from skimage.io import imread, imsave
from skimage.measure import compare_ssim

import numpy as np
import pandas as pd

from tqdm import tqdm
import re

def load_generated_images(images_folder):
    generated_images = []
    names = []
    for img_name in os.listdir(images_folder):
        img = imread(os.path.join(images_folder, img_name))
        generated_images.append(img)

        # m = re.match(r'([A-Za-z0-9_]*.jpg)_([A-Za-z0-9_]*.jpg)', img_name)
        # fr = m.groups()[0]
        # to = m.groups()[1]
        # names.append([fr, to])

    return generated_images

def test():
    args = cmd.args()
    print ("Loading images...")
    generated_images = load_generated_images(args.generated_images_dir)

    print ("Compute inception score...")
    inception_score = get_inception_score(generated_images)
    print ("Inception score %s" % inception_score[0])

    return inception_score


if __name__ == "__main__":
    test()
