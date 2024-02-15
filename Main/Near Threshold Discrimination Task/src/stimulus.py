from dataclasses import dataclass

from psychopy import visual


@dataclass
class Stimulus:
    """
    A complete stimulus for usage in the experiment.
    """
    image_stim: visual.ImageStim
    shape_stim: visual.Polygon
    circle_stim: visual.ShapeStim
    x_stim: visual.ShapeStim
