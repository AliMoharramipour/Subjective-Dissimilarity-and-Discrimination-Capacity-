from psychopy import visual

from stimulus import Stimulus


class ImageStimClickable:
    def __init__(self, stim: Stimulus):
        self.image_stim = stim.image_stim
        self.shape_stim = stim.shape_stim
        self.circle_stim = stim.circle_stim
        self.x_stim = stim.x_stim
        self.stims = [self.image_stim, self.shape_stim, self.circle_stim, self.x_stim]

    def draw(self) -> None:
        for stim in self.stims:
            stim.draw()

    def set_position(self, x: float, y: float) -> None:
        for stim in self.stims:
            stim.pos = x, y

    def set_size(self, size: float) -> None:
        for stim in self.stims:
            stim.size = size
        self.circle_stim.size *= 0.9
