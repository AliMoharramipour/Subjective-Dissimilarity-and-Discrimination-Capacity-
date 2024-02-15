from typing import List

from psychopy import visual

from experiment.stimulus_image import StimulusImage
from util.psychopy_utils import get_keypress, shutdown


class ExperimentArtist:
    """
    Draws the experiment.
    """
    default_font = "Hiragino Maru Gothic Pro"

    response_x = 0.0
    response_y = 0.215
    response_size = 0.4

    candidate_spacing_ratio = 0.6
    leftmost_candidate_x = -0.53
    candidate_spacing_distance = 0.2
    candidate_y = -0.3
    candidate_size = 0.3

    def __init__(self, win: visual.Window):
        self.win = win
        self.trial_instruction_text = visual.TextStim(
            self.win,
            text="上の顔に比べ、以下の顔は似ているものから順に、\nすべての顔が消えるまでクリックしてください。",
            name="instructions-to-subject-in-test-window",
            font=self.default_font,
            color="white",
            pos=(0.0, -0.055),
            units="height",
            height=0.035
        )
        self.loading_text = visual.TextStim(
            self.win,
            text="読み込み中。。。",
            name="loading-text",
            font=self.default_font,
            color="white",
            italic=True,
            pos=(0.0, -0.0),
            units="height",
            height=0.03,
            wrapWidth=0.8,
        )
        self.thank_you_text = visual.TextStim(
            self.win, "お疲れ様です！　ご参加ありがとうございました。\n\n"
                      "これで実験は終了です。\n\n"
                      "実験を終了するには、いずれかのキーを押してください。",
            font=self.default_font,
            units="height",
            height=0.04,
            color="white")

    def draw_bg_color(self, color: list):
        self.win.color = color
        self.win.flip()
        self.win.flip()

    def set_response_image_stim_params(self, stim: StimulusImage):
        stim.image.pos = (self.response_x, self.response_y)
        stim.image.size = self.response_size
        stim.shape.pos = (self.response_x, self.response_y)
        stim.shape.size = self.response_size

    def set_candidate_image_stim_params(self, candidates: List[StimulusImage]):
        candidate_visual_spacing = self.candidate_spacing_ratio / (len(candidates))
        current_candidate_pos = self.leftmost_candidate_x
        # draw the candidate images once
        for c in candidates:
            c.image.pos = (current_candidate_pos, self.candidate_y)
            c.image.size = self.candidate_size  # make this global variable later
            c.shape.pos = (current_candidate_pos, self.candidate_y)
            c.shape.size = self.candidate_size
            current_candidate_pos += candidate_visual_spacing + self.candidate_spacing_distance

    def draw_trial(self, response: StimulusImage, candidates: List[StimulusImage], c_idxs: List[int]):
        # print(f"RESPONSE IMAGE: POS: {response.image.pos}, SIZE: {response.image.size}")

        response.image.draw()
        self.trial_instruction_text.draw()
        for idx in c_idxs:
            """
            print(f"CANDIDATE IMAGE: INDEX: {idx}, "
                  f"GLOBAL INDEX: {candidates[idx].global_idx}, "
                  f"POS: {candidates[idx].image.pos}, "
                  f"SIZE: {candidates[idx].image.size}")
            """
            candidates[idx].image.draw()
            candidates[idx].shape.draw()

        self.win.flip()

    def draw_loading_screen(self):
        self.loading_text.draw()
        self.win.flip()

    def draw_break_screen(self):
        self.win.color = "MidnightBlue"
        self.win.flip()  # need to two flips to change window color

        take_a_break_text = visual.TextBox2(self.win,
                                            "Take a short break. Press any key when you're ready to continue.",
                                            font=self.default_font,
                                            color="white")
        take_a_break_text.draw()

        self.win.flip()

    def draw_thank_you_screen(self):
        self.win.color = "black"
        self.win.flip()  # need to two flips to change window color

        self.thank_you_text.draw()

        self.win.flip()

    def draw_and_wait_on_break_screen(self):
        self.draw_break_screen()
        while True:
            key = get_keypress()
            if key == "escape":
                shutdown(self.win)
            elif key is not None:
                break

    def draw_and_wait_on_thank_you_screen(self):
        self.draw_thank_you_screen()
        while True:
            key = get_keypress()
            if key is not None:
                shutdown(self.win)
