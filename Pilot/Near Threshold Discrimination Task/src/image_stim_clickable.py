from psychopy import visual


class ImageStimClickable:
    def __init__(self, image_stim: visual.ImageStim, shape_stim: visual.Polygon):
        self.image_stim = image_stim
        self.shape_stim = shape_stim

    def draw(self) -> None:
        self.image_stim.draw()
        self.shape_stim.draw()

    def set_position(self, x: float, y: float) -> None:
        self.image_stim.pos = x, y
        self.shape_stim.pos = x, y

    def set_size(self, size: float) -> None:
        self.image_stim.size = size
        self.shape_stim.size = size
