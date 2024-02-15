"""
Helper functions for experiment initialization.
"""
import os
from typing import List

import numpy as np
from psychopy import visual

from experiment.stimulus_image import StimulusImage


def create_image_similarity_matrix(num_rows: int) -> np.array:
    img_similarities = [[0.5 for _ in range(num_rows)] for _ in range(num_rows)]
    return np.array(img_similarities)


def create_stimulus_images(img_paths: List[str], win: visual.Window) -> List[StimulusImage]:
    stims = []
    for idx, img_path in enumerate(img_paths):
        stims.append(StimulusImage(win, idx, image=img_path))
    return stims


def get_img_paths() -> list:
    images = []
    print(os.getcwd())
    img_dir = "../resources/faces"

    for filename in os.listdir(img_dir):
        if filename.endswith(".png"):
            f = os.path.join(img_dir, filename)
            if os.path.isfile(f):
                images.append(f)

    return images
