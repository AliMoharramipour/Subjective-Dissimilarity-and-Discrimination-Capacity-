from typing import List

from stimulus_image_factory import StimulusImageFactory


class MorphSet:
    """
    Tracks the morphs between two StimulusImages
    """
    def __init__(self, start_face: StimulusImageFactory, target_face: StimulusImageFactory, morphs: List[StimulusImageFactory]):
        self.start_face = start_face
        self.target_face = target_face
        self.morphs = morphs
        self.id = f"{start_face.id}-{target_face.id}"
