from typing import List

from stimulus_image import StimulusImage


class MorphSet:
    """
    Tracks the morphs between two StimulusImages
    """
    def __init__(self, start_face: StimulusImage, target_face: StimulusImage, morphs: List[StimulusImage]):
        self.start_face = start_face
        self.target_face = target_face
        self.morphs = morphs
        self.id = f"{start_face.id}-{target_face.id}"
