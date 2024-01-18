# -*- coding: utf-8 -*-

import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import os
from typing import List, Tuple, Dict

import pandas as pd
import psychopy
from psychopy import visual, data, event, logging, core, gui

import selection_algorithms_sk_mds
from infotuple_py3 import body_metrics
from matrix_utils import get_output_matrix_from_embedding
from oracle_response import OracleResponse

default_font = "Hiragino Maru Gothic Pro"


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


def create_image_similarity_matrix(num_rows: int) -> np.array:
    img_similarities = [[0.5 for _ in range(num_rows)] for _ in range(num_rows)]
    return np.array(img_similarities)


def create_stimulus_images(img_paths: List[str], win: visual.Window) -> List[StimulusImage]:
    stims = []
    for idx, img_path in enumerate(img_paths):
        stims.append(StimulusImage(win, idx, image=img_path))
    return stims


def get_img_paths() -> list:
    images = []
    img_dir = "../resources/faces"

    for filename in os.listdir(img_dir):
        if filename.endswith(".png"):
            f = os.path.join(img_dir, filename)
            if os.path.isfile(f):
                images.append(f)

    return images


def is_mouse_pressed(test_mouse: event.Mouse):
    for button in test_mouse.getPressed():
        if button != 0:
            return True
    return False


def get_keypress():
    keys = event.getKeys()
    if keys:
        return keys[0]
    else:
        return None


def shutdown(win: visual.Window):
    win.close()
    core.quit()


class SubjectOracle:
    click_data = []

    def __init__(self, win: visual.Window,
                 test_mouse: event.Mouse,
                 global_idx_to_stim: Dict[int, StimulusImage],
                 log_output_dir: str):
        self.win = win
        self.test_mouse = test_mouse
        self.global_idx_to_stim = global_idx_to_stim
        self.num_tasks_done = 0
        self.log_output_dir = log_output_dir

    def click_logging(self):
        log_fields = ["iteration", "trial", "click_pos", "click_time", "clicked_stim",
                      "clicked_stim_pos", "test_stims", "target_stim"]
        df = pd.DataFrame(log_fields)

    def oracle(self, candidate_tuple: Tuple[int]) -> OracleResponse:
        print("\n\n================= NEW ROUND ====================")
        print(f"INDEX TO STIM: {self.global_idx_to_stim}")
        print(f"Working with tuple {candidate_tuple}")
        response_list = [candidate_tuple[0]]
        response_image_stim = self.global_idx_to_stim[candidate_tuple[0]]
        candidates = [self.global_idx_to_stim[global_idx] for global_idx in candidate_tuple[1:]]
        print(f"Candidates: {[c.global_idx for c in candidates]}")

        click_data = []

        self.win.color = [0, 0, 0]
        self.win.flip()
        self.win.flip()

        # ranking order from closest to farthest (0th position is closest to response)
        # draw the response
        response_image_stim.image.pos = (0.0, 0.215)
        response_image_stim.image.size = 0.4
        response_image_stim.shape.pos = (0.0, 0.215)
        response_image_stim.shape.size = 0.4

        # instructions
        text_stim = visual.TextStim(
            self.win,
            text="上の顔に比べ、以下の顔は似ているものから順に、\nすべての顔が消えるまでクリックしてください。",
            name="instructions-to-subject-in-test-window",
            font=default_font,
            color="white",
            pos=(0.0, -0.055),
            units="height",
            height=0.035
        )

        candidate_visual_spacing = 0.6 / (len(candidates))
        current_candidate_pos = -0.53
        # draw the candidate images once
        for c in candidates:
            c.image.pos = (current_candidate_pos, -0.3)
            c.image.size = 0.3  # make this global variable later
            c.shape.pos = (current_candidate_pos, -0.3)
            c.shape.size = 0.3
            current_candidate_pos += candidate_visual_spacing + 0.2

        c_idxs = [idx for idx in range(len(candidates))]

        while c_idxs:
            print(f"RESPONSE IMAGE: POS: {response_image_stim.image.pos}, SIZE: {response_image_stim.image.size}")
            self.win.flip()
            response_image_stim.image.draw()
            text_stim.draw()
            for idx in c_idxs:
                print(f"CANDIDATE IMAGE: INDEX: {idx}, "
                      f"GLOBAL INDEX: {candidates[idx].global_idx}, "
                      f"POS: {candidates[idx].image.pos}, "
                      f"SIZE: {candidates[idx].image.size}")
                candidates[idx].image.draw()
                candidates[idx].shape.draw()

            self.win.flip()

            while True:
                clicked = False

                key = get_keypress()
                if key == "escape":
                    shutdown(self.win)
                else:
                    for idx, c_idx in enumerate(c_idxs):
                        c = candidates[c_idx]
                        if self.test_mouse.isPressedIn(c.shape):
                            click_pos = self.test_mouse.getPos()
                            click_time = core.monotonicClock.getTime()

                            print(f"Mouse clicked on {c.global_idx}")
                            response_list.append(c.global_idx)
                            c_idxs.pop(idx)
                            clicked = True

                            # this ONLY works if we keep oracle in the main script because lowercase m monotonicClock
                            # (not MonotonicClock) starts counting from the time psychopy.core is imported
                            click_data.append({"trial": self.num_tasks_done,
                                               "click_time": click_time,
                                               "click_pos": click_pos,
                                               "clicked_stim": c.global_idx,
                                               "clicked_stim_pos": idx,
                                               "candidate_stims": candidate_tuple[1:],
                                               "target_stim": candidate_tuple[0]})

                            while is_mouse_pressed(self.test_mouse):
                                # wait for mouse to be unpressed
                                pass
                            break  # very necessary to prevent weird for loop errors on pop
                    if clicked:
                        break

        self.num_tasks_done += 1
        self.win.flip()

        loading_text_stim = visual.TextStim(
            self.win,
            text="読み込み中。。。",
            name="loading-text",
            font=default_font,
            color="white",
            italic=True,
            pos=(0.0, -0.0),
            units="height",
            height=0.03,
            wrapWidth=0.8,
        )

        loading_text_stim.draw()
        self.win.flip()

        if self.num_tasks_done % 90 == 0:
            draw_take_a_break(self.win)

        print(f"responding with: {response_list}")
        return OracleResponse(tuple(response_list), click_data)

    def get_oracle(self):
        return self.oracle


def draw_take_a_break(win: visual.Window):
    win.color = "MidnightBlue"
    win.flip()
    win.flip()  # need to two flips to change window color

    take_a_break_text = visual.TextBox2(win, "Take a short break. Press any key when you're ready to continue.",
                                        font=default_font,
                                        color="white")
    take_a_break_text.draw()

    win.flip()

    while True:
        key = get_keypress()
        if key == "escape":
            shutdown(win)
        elif key is not None:
            break


def draw_thank_you(win: visual.Window):
    win.color = "black"
    win.flip()
    win.flip()  # need to two flips to change window color

    thank_you_text = visual.TextStim(win, "お疲れ様です！　ご参加ありがとうございました。\n\n"
                                          "これで実験は終了です。\n\n"
                                          "実験を終了するには、いずれかのキーを押してください。",
                                     font=default_font,
                                     units="height",
                                     height=0.04,
                                     color="white")
    thank_you_text.draw()

    win.flip()

    while True:
        key = get_keypress()
        if key is not None:
            shutdown(win)


def main():
    # Ensure that relative paths start from the same directory as this script
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(_this_dir)
    # Store info about the experiment session
    psychopy_version = f"{psychopy.__version__}"
    exp_name = "Face Similarity experiment"

    exp_info = {
        "participant": "名前を入力してください。",
        "session": "1",
    }
    # --- Show participant info dialog --
    dlg = gui.DlgFromDict(dictionary=exp_info, sortKeys=False, title=exp_name)
    if not dlg.OK:
        core.quit()  # user pressed cancel
    exp_info["date"] = data.getDateStr()  # add a simple timestamp
    exp_info["expName"] = exp_name
    exp_info["psychopy_version"] = psychopy_version

    run_label = f"{datetime.now().strftime('%Y.%m.%d-%H.%M.%S')}-{exp_info['participant']}-{exp_info['session']}"
    output_dir = f"../output/runs/{run_label}"
    os.makedirs(output_dir)

    # An ExperimentHandler isn't essential but helps with data saving
    this_exp = data.ExperimentHandler(name=exp_name, version=psychopy_version,
                                      extraInfo=exp_info, runtimeInfo=None,
                                      savePickle=True, saveWideText=True,
                                      dataFileName=f"{output_dir}/psychopy.log")
    # save a log file for detail verbose info
    warning_log_file = logging.LogFile(f"{output_dir}/warn.log", level=logging.WARNING)
    info_log_file = logging.LogFile(f"{output_dir}/info.log", level=logging.INFO, filemode="w")

    logging.console.setLevel(logging.WARNING)

    win = visual.Window([1920, 1080], units="height", screen=0, fullscr=True)
    loading_text_stim = visual.TextStim(
        win,
        text="読み込み中。。。",
        name="loading-text",
        color="white",
        font=default_font,
        italic=True,
        pos=(0.0, -0.0),
        units="height",
        height=0.03,
        wrapWidth=0.8,
    )
    loading_text_stim.autoDraw = True
    loading_text_stim.draw()
    win.mouseVisible = True

    # store frame rate of monitor if we can measure it
    exp_info["frameRate"] = win.getActualFrameRate()
    if exp_info["frameRate"] is not None:
        frame_dur = 1.0 / round(exp_info["frameRate"])
    else:
        frame_dur = 1.0 / 60.0  # could not measure, so guess
    # --- Setup input devices ---
    experiment_mouse = event.Mouse(win)

    win.flip()

    # --- Run the experiment ---
    n_burn_ins = 1
    n_components = 5
    tuple_size = 5
    num_iterations = 9
    stim_list = create_stimulus_images(get_img_paths(), win)
    image_similarity_matrix = create_image_similarity_matrix(len(stim_list))  # numImages == (len(stim_list)^2)/2
    matrix_size = len(image_similarity_matrix)
    global_idx_to_stim = {}
    for stim in stim_list:
        global_idx_to_stim[stim.global_idx] = stim

    run_data_list = []
    click_data_list = []

    run_data = {"matrix_size": matrix_size, "tuple_size": tuple_size, "n_components": n_components}

    loading_text_stim.autoDraw = False

    selection_algo = selection_algorithms_sk_mds.SelectionAlgorithm(image_similarity_matrix,
                                                                    n_burn_ins,
                                                                    num_iterations,
                                                                    SubjectOracle(win, experiment_mouse,
                                                                                  global_idx_to_stim,
                                                                                  output_dir)
                                                                    .get_oracle(),
                                                                    body_metrics.primal_body_selector,
                                                                    tuple_size,
                                                                    verbose=False)
    output_embedding, dissim_matrix = selection_algo.selection_algorithm(run_data_list, run_data, click_data_list, [],
                                                                         n_components=n_components)

    dissim_matrix_from_embedding = get_output_matrix_from_embedding(output_embedding, matrix_size)

    graph_dataframe = pd.DataFrame(run_data_list)
    graph_dataframe["n_components"] = graph_dataframe["n_components"].astype("category")
    graph_dataframe.to_csv(f"{output_dir}/run_data.csv")

    click_dataframe = pd.DataFrame(click_data_list)
    click_dataframe.to_csv(f"{output_dir}/click_data.csv")

    dissim_matrix_df = pd.DataFrame(data=dissim_matrix.astype(float))
    dissim_matrix_df.to_csv(f"{output_dir}/dissim_matrix.csv", sep=" ", header=False, float_format="%.6f",
                            index=False)

    output_embedding_df = pd.DataFrame(data=output_embedding.astype(float))
    output_embedding_df.to_csv(f"{output_dir}/output_embedding.csv", sep=" ", header=False, float_format="%.6f",
                               index=False)

    dm_from_embedding = pd.DataFrame(data=dissim_matrix_from_embedding.astype(float))
    dm_from_embedding.to_csv(f"{output_dir}/dissim_matrix_from_embedding.csv", sep=" ", header=False,
                             float_format="%.6f", index=False)

    print(f"Output dissimilarity matrix:\n{dissim_matrix_from_embedding}")
    print(f"Wall clock total program time: {core.monotonicClock.getTime()}")
    logging.flush()

    draw_thank_you(win)


if __name__ == "__main__":
    main()

    """
    win = visual.Window([1920, 1080], units="height", screen=0, fullscr=False)
    draw_thank_you(win)
    """
