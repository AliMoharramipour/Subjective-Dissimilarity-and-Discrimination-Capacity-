import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict, List

from psychopy import visual

from stimulus_image import StimulusImage


class ExperimentResourceFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_image_info_from_filepath(img_filepath: str) -> Tuple[int, int, int]:
        start_face_id, target_face_id, morph_id = os.path.splitext(os.path.split(img_filepath)[1])[0].split("-")
        return int(start_face_id), int(target_face_id), int(morph_id)

    @staticmethod
    def create_morph_sets(img_dir: str) -> Dict[str, Dict[int, StimulusImage]]:
        """
        Could make this text files that list out the start-target pairs and number of morphs and we just read the text.
        As opposed to scanning the actual files.
        morph between face 0 and face 1, for example. so we could have 0-1.500.jpg, or 291-500.999.jpg.
        291-500.1000.jpg == 500.jpg && 291-500.0.jpg == 291.jpg
        :param img_dir:
        :return:
        """
        # filenames are 0-50.300.png for example.
        morph_sets = defaultdict(dict)  # heap so that the list is from 0 morph to 999 morph always
        image_names = get_img_paths(img_dir)

        for image_name in image_names:
            start_face_id, target_face_id, morph_id = re.split("[-.]",
                                                               os.path.splitext(os.path.split(image_name)[1])[0])
            morph_sets[f"{start_face_id}-{target_face_id}"][int(morph_id)] = \
                StimulusImage(f"{start_face_id}-{target_face_id}-{morph_id}", start_face_id, target_face_id,
                              morph_id, image_name)
        return morph_sets

    def create_stimulus_image(self, img_filepath: str, img_id: str) -> StimulusImage:
        start_face_id, target_face_id, morph_id = self.get_image_info_from_filepath(img_filepath)
        return StimulusImage(img_id, start_face_id, target_face_id, morph_id, img_filepath)


def get_img_paths(img_dir: str) -> list:
    images = []

    for filepath in Path(img_dir).rglob("*.png"):
        images.append(str(filepath))

    return images
