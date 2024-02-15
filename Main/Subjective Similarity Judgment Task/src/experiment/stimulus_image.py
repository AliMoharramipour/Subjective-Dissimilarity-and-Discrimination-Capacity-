from psychopy import visual


class StimulusImage:
    img_size = 0.2

    def __init__(self, win: visual.Window, global_idx: int, image: str):
        self.global_idx = global_idx  # can"t use ID as python reserves that for object id
        self.q_idx = -1  # set for each individual question
        self.name = f"face-{global_idx}"
        self.image = visual.ImageStim(win, image, units="height", size=self.img_size, name=f"{self.name}-image",
                                      interpolate=True, texRes=512)
        # default to face picture size from face generator paper
        self.shape = visual.Polygon(win, edges=99, units="height", size=self.img_size, name=f"{self.name}-shape",
                                    opacity=0.0, fillColor="red", lineColor="red", color="red")
