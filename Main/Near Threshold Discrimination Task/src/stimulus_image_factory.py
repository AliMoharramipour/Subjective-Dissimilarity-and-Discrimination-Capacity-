from enum import Enum
from typing import Dict

from psychopy import visual
from psychopy.visual.basevisual import MinimalStim

from stimulus import Stimulus


class StimulusImageFactory:
    img_size = 0.2
    tex_res = 512

    def __init__(self, stim_id: str, start_face_id: int, target_face_id: int,
                 morph_id: int, img_filepath: str):
        self.id = f"{stim_id}"
        self.morph_id = morph_id  # between 0 and 999, where 0 is the start face and 999 is the target face
        self.start_face_id = start_face_id
        self.target_face_id = target_face_id
        self.img_filepath = img_filepath

    def create_image_stim(self, win: visual.Window, dup=0) -> visual.ImageStim:
        return visual.ImageStim(win, self.img_filepath, units="height", size=self.img_size,
                                name=f"{self.id}_{dup}_image", interpolate=True, texRes=512)

    def create_shape_stim(self, win: visual.Window, dup=0) -> visual.Polygon:
        return visual.Polygon(win, edges=99, units="height", size=self.img_size, name=f"{self.id}_{dup}_shape",
                              opacity=0.0, fillColor="#ff6565", lineColor="#ff6565")

    def create_circle_stim(self, win: visual.Window, dup=0) -> visual.ShapeStim:
        return visual.ShapeStim(win, vertices="circle", units="height", size=self.img_size, lineWidth=8,
                                name=f"{self.id}_{dup}_shape", opacity=0.0, fillColor=None, lineColor="#42ff25")

    def create_x_stim(self, win: visual.Window, dup=0) -> visual.ShapeStim:
        scale = 0.5
        cross_vertices = [
            (-0.1*scale, +0.5),  # up
            (+0.1*scale, +0.5),
            (+0.1*scale, +0.1*scale),
            (+0.5, +0.1*scale),  # right
            (+0.5, -0.1*scale),
            (+0.1*scale, -0.1*scale),
            (+0.1*scale, -0.5),  # down
            (-0.1*scale, -0.5),
            (-0.1*scale, -0.1*scale),
            (-0.5, -0.1*scale),  # left
            (-0.5, +0.1*scale),
            (-0.1*scale, +0.1*scale),
        ]
        return visual.ShapeStim(win, vertices=cross_vertices, ori=45.0, units="height", size=self.img_size,
                                name=f"{self.id}_{dup}_shape", opacity=0.0, fillColor="red", lineColor="red",
                                color="red")

    def create_stim(self, win: visual.Window, dup=0):
        return Stimulus(
            self.create_image_stim(win, dup),
            self.create_shape_stim(win, dup),
            self.create_circle_stim(win, dup),
            self.create_x_stim(win, dup)
        )
