from psychopy import visual


class StimulusImage:
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
                              opacity=0.0, fillColor="red", lineColor="red", color="red")
