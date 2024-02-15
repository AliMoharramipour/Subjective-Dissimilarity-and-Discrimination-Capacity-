# -*- coding: utf-8 -*-

import os
from datetime import datetime

import pandas as pd
import psychopy
from psychopy import visual, data, event, logging, core, gui

from experiment import selection_algorithms_sk_mds
from experiment.experiment_artist import ExperimentArtist
from experiment.experiment_init import create_image_similarity_matrix, create_stimulus_images, get_img_paths
from logger.log_writer import LogWriter
from util.matrix_utils import get_output_matrix_from_embedding
from experiment.oracle_runner import OracleRunner


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

    win = visual.Window([1280, 720], units="height", screen=0, fullscr=True)
    experiment_artist = ExperimentArtist(win)
    experiment_artist.loading_text.autoDraw = True
    experiment_artist.loading_text.draw()
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

    log_writer = LogWriter(output_dir)

    experiment_artist.loading_text.autoDraw = False

    selection_algo = selection_algorithms_sk_mds.SelectionAlgorithm(image_similarity_matrix,
                                                                    n_burn_ins,
                                                                    num_iterations,
                                                                    OracleRunner(win, experiment_artist,
                                                                                 experiment_mouse,
                                                                                 global_idx_to_stim,
                                                                                 output_dir)
                                                                    .get_oracle(),
                                                                    log_writer,
                                                                    tuple_size,
                                                                    verbose=False)
    output_embedding, dissim_matrix = selection_algo.selection_algorithm(run_data_list, run_data, click_data_list, [],
                                                                         n_components=n_components)

    dissim_matrix_from_embedding = get_output_matrix_from_embedding(output_embedding, matrix_size)

    log_writer.write_all(click_data_list, run_data_list, output_embedding, dissim_matrix, matrix_size)

    print(f"Output dissimilarity matrix:\n{dissim_matrix_from_embedding}")
    print(f"Wall clock total program time: {core.monotonicClock.getTime()}")
    logging.flush()

    experiment_artist.draw_and_wait_on_thank_you_screen()


if __name__ == "__main__":
    main()

    """
    win = visual.Window([1920, 1080], units="height", screen=0, fullscr=False)
    draw_thank_you(win)
    """
