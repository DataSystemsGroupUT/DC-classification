#!/usr/bin/env python3
"""Extract deep CNN features from a set of images and dump them as Numpy arrays image_file_name.npy"""

import argparse
import numpy as np
import cv2
from scipy import ndimage
from os.path import basename, join, exists
from os import makedirs
from threaded_generator import threaded_generator
from time import time
import sys
np.random.seed(13)
PATCH_SIZES = [224]
SCALES = [1]
DEFAULT_INPUT_DIR = "/home/ubuntu/DCGAN_600epochs/1500_epochs/1500epochs/" 
DEFAULT_PREPROCESSED_ROOT = "/home/ubuntu/preprocessing/maincode/files/features_over_epochs/0_fakes_1500epochs/" 
PATCHES_PER_IMAGE = 5
AUGMENTATIONS_PER_IMAGE = 1
COLOR_LO = 0.7
COLOR_HI = 1.3
BATCH_SIZE = 16     # decrease if necessary
NUM_CACHED = 160
def recursive_glob(root_dir, file_template="*.png"): # change your input image format here
    """Traverse directory recursively. Starting with Python version 3.5, the glob module supports the "**" directive"""

    if sys.version_info[0] * 10 + sys.version_info[1] < 35:
        import fnmatch
        import os
        matches = []
        for root, dirnames, filenames in os.walk(root_dir):
            for filename in fnmatch.filter(filenames, file_template):
                matches.append(os.path.join(root, filename))
        return matches
    else:
        import glob
        return glob.glob(root_dir + "/**/" + file_template, recursive=True)


def get_crops(img, size, n, seed=None):
    """Creates random square crops of given size from a Numpy image array. No rotation added

    # Arguments
        img: Numpy image array.
        size: size of crops.
        n: number of crops
        seed: Random seed.
    # Returns
        Numpy array of crops, shape (n, size, size, c).
    """
    h, w, c = img.shape
    assert all([size < h, size < w])
    crops = []
    for _ in range(n):
        top = np.random.randint(low=0, high=h - size + 1)
        left = np.random.randint(low=0, high=w - size + 1)
        crop = img[top: top + size, left: left + size].copy()
        crop = np.rot90(crop, np.random.randint(low=0, high=4))
        if np.random.random() > 0.5:
            crop = np.flipud(crop)
        if np.random.random() > 0.5:
            crop = np.fliplr(crop)
        crops.append(crop)
    
    crops = np.stack(crops)
    assert crops.shape == (n, size, size, c)
    print('shapeofcrop',crops.shape)
    return crops

def norm_pool(features, p=3):
    """Performs descriptor pooling

    # Arguments
        features: Numpy array of descriptors.
        p: degree of pooling.
    # Returns
        Numpy array of pooled descriptor.
    """
    return np.power(np.power(features, p).mean(axis=0), 1/p)


def encode(crops, model):
    """Encodes crops

    # Arguments
        crops: Numpy array of crops.
        model: Keras encoder.
    # Returns
        Numpy array of pooled descriptor.
    """
    features = model.predict(crops)
    pooled_features = norm_pool(features)
    return pooled_features

def process_image(image_file):
    """Extract multiple crops from a single image

    # Arguments
        image_file: Path to image.
    # Yields
        Numpy array of image crops.
    """
    img = cv2.imread(image_file)
    img = cv2.resize(img, (300, 300)) 
    img= cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img=np.array(img,dtype='float64')
    if SCALE != 1:
        img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    for _ in range(AUGMENTATIONS_PER_IMAGE):
        single_image_crops = get_crops(img, PATCH_SZ, PATCHES_PER_IMAGE)
        yield single_image_crops


def crops_gen(file_list):
    """Generates batches of crops from image list, one augmentation a time

    # Arguments
        file_list: List of image files.
    # Yields
        Tuple of Numpy array of image crops and name of the file.
    """
    for i, (image_file, output_file) in enumerate(file_list):
        print("Crops generator:", i + 1)
        for crops in process_image(image_file):
            yield crops, output_file
    


def features_gen(crops_and_output_file, model):
    """Processes crop generator, encodes them and dumps pooled descriptors

    # Arguments
        crops_and_output_file: generator of crops and file names.
        model: Keras encoder.
    # Returns: None
    """
    ts = time()
    current_file = None
    pooled_features = []
    i = 0
    for j, (crops, output_file) in enumerate(crops_and_output_file):
        if current_file is None:
            current_file = output_file
        features = encode(crops, model)
        if output_file == current_file:
            pooled_features.append(features)
        else:
            np.save(current_file, np.stack(pooled_features))
            pooled_features = [features]
            current_file = output_file
            average_time = int((time() - ts) / (i + 1))
            print("Feature generator: {}, {} sec/image.".format(i + 1, average_time))
            i += 1
    if len(pooled_features) > 0:
        np.save(current_file, np.stack(pooled_features))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--images",
        required=False,
        default=DEFAULT_INPUT_DIR,
        metavar="img_dir",
        help="Input image directory.")
    arg("--features",
        required=False,
        default=DEFAULT_PREPROCESSED_ROOT,
        metavar="feat_dir",
        help="Feature root dir.")
    args = parser.parse_args()
    INPUT_DIR = args.images
    PREPROCESSED_ROOT = args.features
    from models import ResNet, Inception, VGG
    NN_MODELS = [VGG]
    print('input_directory', INPUT_DIR)
    input_files = recursive_glob(INPUT_DIR)
    for SCALE in SCALES:
        print("SCALE:", SCALE)
        for NN_MODEL in NN_MODELS:
            print("NN_MODEL:", NN_MODEL.__name__)
            for PATCH_SZ in PATCH_SIZES:
                print("PATCH_SZ:", PATCH_SZ)
                PREPROCESSED_PATH = join(PREPROCESSED_ROOT, "{}-{}-{}".format(NN_MODEL.__name__, SCALE, PATCH_SZ))
                if not exists(PREPROCESSED_PATH):
                    makedirs(PREPROCESSED_PATH)
                model = NN_MODEL(batch_size=BATCH_SIZE)
                output_files = [join(PREPROCESSED_PATH, basename(f).replace("png", "npy")) for f in input_files]
                file_list = zip(input_files, output_files)
                crops_and_output_file = crops_gen(file_list)
                crops_and_output_file_ = threaded_generator(crops_and_output_file, num_cached=NUM_CACHED)
                features_gen(crops_and_output_file_, model)
