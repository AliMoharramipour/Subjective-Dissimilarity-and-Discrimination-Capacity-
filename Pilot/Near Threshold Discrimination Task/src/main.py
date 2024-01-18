import copy
import os
from datetime import datetime

import numpy as np
import pandas as pd
import psychopy
from numpy.random import shuffle
from psychopy import visual, data, event, core, gui, logging
from psychopy.event import Mouse

from experiment_resource_handler import ExperimentResourceFactory
from image_stim_clickable import ImageStimClickable
from stair_handler_with_custom_logging import StairHandlerWithCustomLogging

default_font = "Hiragino Maru Gothic Pro"


def get_keypress():
    keys = event.getKeys()
    if keys:
        return keys[0]
    else:
        return None


def shutdown(win: visual.Window):
    win.close()
    core.quit()


def is_mouse_pressed(test_mouse: event.Mouse):
    for button in test_mouse.getPressed():
        if button != 0:
            return True
    return False


def draw_take_a_break(win: visual.Window):
    #win.color = "MidnightBlue"
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
    # assign each face a number from 0 to 100
    # create some info to store with the data
    # show face n, face n, face n+k. k changes with the reversal so k is stepSize
    face_pair = input("Please enter the face pair to test as face_id-face_id, i.e. 2-99 for face 2 and face 99. If you "
                      "want interleaved staircases on all pairs just press enter: \n -> ")
    if face_pair == "":
        resource_dir = "resources/face_pairs"
    else:
        resource_dir = f"resources/face_pairs/{face_pair}"
    min_face_num = 1
    max_face_num = 1000

    exp_name = "Face JND Experiment"

    exp_info = {
        "participant": "名前を入力してください。",
        "session": "1"
    }

    # present a dialogue to change params
    dlg = gui.DlgFromDict(exp_info, title=exp_name, fixed=["date"])
    if not dlg.OK:
        core.quit()  # the user hit cancel so exit
    exp_info["date"] = data.getDateStr()  # add a simple timestamp
    exp_info["expName"] = exp_name
    exp_info["psychopy_version"] = f"{psychopy.__version__}"

    run_label = f"{datetime.now().strftime('%Y.%m.%d-%H.%M.%S')}-{exp_info['participant']}-{exp_info['session']}"
    output_dir = f"output/runs/{run_label}"
    os.makedirs(output_dir)

    log_format = ["exp_trial_num", "staircase_id", "staircase_trial_num", "intensity", "current_step_size",
                  "clicked_face_id", "clicked_face_position", "click_time_experiment", "click_time_current_trial",
                  "faces_shown", "was_correct", "next_direction"]
    warning_log_file = logging.LogFile(f"{output_dir}/warn.log", level=logging.WARNING, filemode="w")
    info_log_file = logging.LogFile(f"{output_dir}/info.log", level=logging.INFO, filemode="w")

    logging.console.setLevel(logging.WARNING)

    # create window and stimuli
    win = visual.Window([1920, 1080], units='height', screen=0, fullscr=True)
    instruction_text = visual.TextStim(win, pos=[0, -0.3], height=0.03,
                                       text="３つの顔のうち、似ていない顔をクリックしてください。",
                                       name="instruction-text",
                                       font="Hiragino Maru Gothic Pro")
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
    test_mouse: Mouse = event.Mouse(win)

    win.flip()

    # create all the face_pairs and morphs
    # the morph set is a map of "start_face_id-target_face_id" to map of "morph" to StimulusImage
    # where [1] is start_face and [1000] is target face
    experiment_resources = ExperimentResourceFactory()
    morph_sets = experiment_resources.create_morph_sets(resource_dir)

    # ---------------------
    # create the stimuli
    # ---------------------

    # create staircases where each startVal is the startFace and targetFace is in extraInfo.
    # modulate steps between start and target using the given file morphs organized by filename - 0-1.0 for 0th face in
    # morph between face 0 and face 1, for example. so we could have 0-1.500.jpg, or 291-500.999.jpg.
    # 291-500.1000.jpg == 500.jpg && 291-500.0.jpg == 291.jpg
    exp_info["n_trials"] = "60"  # need to make this string for staircase logging code later
    max_step = 400
    stairs = []
    for face_pair, _ in morph_sets.items():
        this_info = copy.copy(exp_info)
        this_info["face_pair"] = face_pair
        stair_wrapper = StairHandlerWithCustomLogging(
            data.StairHandler(startVal=500, extraInfo=this_info, nTrials=60, nUp=1, nDown=2, minVal=1,
                              maxVal=max_face_num - 1, nReversals=11, name=face_pair, stepType="lin",
                              stepSizes=[int(n * max_step)
                                         for n in
                                         [1, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.2, 0.1, 0.1, 0.05]]))
        stairs.append(stair_wrapper)

    bg_colors = ["gray", "navy"]
    cur_color_idx = 0

    log_format = ["exp_trial_num", "staircase_id", "staircase_trial_num", "distance", "current_step_size",
                  "clicked_face_id", "click_position", "click_time_experiment", "click_time_current_trial",
                  "faces_shown", "was_correct", "next_direction"]
    experiment_log_tracker = []

    rng = np.random.default_rng()

    # experiment begins
    experiment_clock = core.MonotonicClock()

    for trial_n in range(int(exp_info["n_trials"])):
        if trial_n != 0 and trial_n % 15 == 0:
            draw_take_a_break(win)

        rng.shuffle(stairs)

        for stair_wrapper in stairs:
            cur_stair = stair_wrapper.stair

            distance = next(cur_stair)

            experiment_log_entry = {"exp_trial_num": trial_n}

            experiment_log_entry["staircase_id"] = cur_stair.name
            experiment_log_entry["staircase_trial_num"] = cur_stair.thisTrialN
            experiment_log_entry["distance"] = distance
            experiment_log_entry["current_step_size"] = cur_stair.stepSizeCurrent

            face_pair = cur_stair.extraInfo['face_pair']

            direction = rng.integers(2)

            if direction == 0:  # go left
                face = rng.integers(distance + 1, max_face_num, endpoint=True)
                face_n_k = ImageStimClickable(morph_sets[f"{face_pair}"][face - distance].create_image_stim(win),
                                              morph_sets[f"{face_pair}"][face - distance].create_shape_stim(win))
                sign_distance = -1
            else:  # go right
                face = rng.integers(min_face_num, max_face_num - distance, endpoint=True)
                face_n_k = ImageStimClickable(morph_sets[f"{face_pair}"][face + distance].create_image_stim(win),
                                              morph_sets[f"{face_pair}"][face + distance].create_shape_stim(win))
                sign_distance = 1

            face_n_0 = ImageStimClickable(morph_sets[f"{face_pair}"][face].create_image_stim(win),
                                          morph_sets[f"{face_pair}"][face].create_shape_stim(win))

            face_n_1 = ImageStimClickable(morph_sets[f"{face_pair}"][face].create_image_stim(win, 1),
                                          morph_sets[f"{face_pair}"][face].create_shape_stim(win, 1))

            print(f"face_pair={face_pair}, morph={face}, dist={distance}")

            positions = [-0.45, 0, 0.45]
            face_stims = [face_n_0, face_n_1, face_n_k]
            rng.shuffle(face_stims)
            for idx, face_stim in enumerate(face_stims):
                face_stim.set_position(positions[idx], 0)
                face_stim.set_size(0.4)
                face_stim.draw()

            instruction_text.draw()

            loading_text_stim.autoDraw = False
            win.flip()

            trial_clock = core.MonotonicClock()
            was_correct = False
            while True:
                if test_mouse.isPressedIn(face_n_0.shape_stim):
                    click_time_experiment = experiment_clock.getTime()
                    click_time_current_trial = trial_clock.getTime()
                    click_position = test_mouse.getPos()

                    clicked_stim_id = face_n_0.image_stim.name
                    break
                if test_mouse.isPressedIn(face_n_1.shape_stim):
                    click_time_experiment = experiment_clock.getTime()
                    click_time_current_trial = trial_clock.getTime()
                    click_position = test_mouse.getPos()

                    clicked_stim_id = face_n_1.image_stim.name
                    break
                if test_mouse.isPressedIn(face_n_k.shape_stim):
                    click_time_experiment = experiment_clock.getTime()
                    click_time_current_trial = trial_clock.getTime()
                    click_position = test_mouse.getPos()

                    clicked_stim_id = face_n_k.image_stim.name
                    was_correct = True
                    break


            experiment_log_entry["clicked_face_id"] = clicked_stim_id
            experiment_log_entry["click_position"] = click_position
            experiment_log_entry["click_time_experiment"] = click_time_experiment
            experiment_log_entry["click_time_current_trial"] = click_time_current_trial
            experiment_log_entry["faces_shown"] = f"{face_stims[0].image_stim.name}," \
                                                  f"{face_stims[1].image_stim.name}," \
                                                  f"{face_stims[2].image_stim.name}"
            experiment_log_entry["was_correct"] = was_correct

            print(f"Was correct?: {was_correct}")
            cur_stair.addResponse(was_correct)  # so that the staircase adjusts itself

            experiment_log_entry["next_direction"] = cur_stair.currentDirection

            # add logging for face_n per round
            stair_wrapper.logger.store_response(face_pair, face, distance, was_correct, [face, face + sign_distance * distance])

            experiment_log_tracker.append(experiment_log_entry)

            loading_text_stim.draw()

            """
            if cur_color_idx == 0:
                cur_color_idx = 1
            else:
                cur_color_idx = 0

            win.color = bg_colors[cur_color_idx]
            """
            win.flip()
            """
            loading_text_stim.draw()
            win.flip()
            """

    for stair_wrapper in stairs:
        stair_wrapper.logger.write_logfile(stair_wrapper.stair, output_dir=output_dir)
        stair_wrapper.write_stair_as_pickle()

    pd.DataFrame(data=experiment_log_tracker).to_csv(f"{output_dir}/experiment.csv")

    logging.flush()

    draw_thank_you(win)
    shutdown(win)


main()
