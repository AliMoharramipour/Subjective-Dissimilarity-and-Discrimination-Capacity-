from enum import Enum

from psychopy import visual


class StimulusType(Enum):
    image_stim: visual.ImageStim
    shape_stim: visual.Polygon
    circle_stim: visual.ShapeStim
    x_stim: visual.ShapeStim
