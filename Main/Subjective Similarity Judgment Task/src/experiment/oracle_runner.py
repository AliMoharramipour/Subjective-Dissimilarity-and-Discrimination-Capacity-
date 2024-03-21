import time
from typing import Dict, Tuple, List, Set

from psychopy import visual, event, core

from experiment.experiment_artist import ExperimentArtist
from experiment.oracle_response import OracleResponse
from experiment.stimulus_image import StimulusImage
from util.psychopy_utils import is_mouse_pressed, get_keypress, shutdown


class OracleRunner:
    # click_data_manager = ClickDataManager()
    click_data = []
    trial_timeout = 30  # in seconds

    def __init__(self, win: visual.Window,
                 experiment_artist: ExperimentArtist,
                 test_mouse: event.Mouse,
                 global_idx_to_stim: Dict[int, StimulusImage],
                 log_output_dir: str):
        self.win = win
        self.test_mouse = test_mouse
        self.global_idx_to_stim = global_idx_to_stim
        self.num_tasks_done = 0
        self.log_output_dir = log_output_dir
        self.experiment_artist = experiment_artist

    def check_stim_clicked(self, candidates: List[StimulusImage], response_list: list, click_data: list,
                           candidate_tuple: Tuple[int], c_idxs: List[int], trial_start_time: float,
                           enforce_timeout: bool) -> bool:
        while True:
            clicked = False
            key = get_keypress()
            current_time = time.time()
            if enforce_timeout and (current_time - trial_start_time > self.trial_timeout):
                return False
            if key == "escape":
                shutdown(self.win)
            else:
                for idx, c_idx in enumerate(c_idxs):
                    c = candidates[c_idx]
                    if self.test_mouse.isPressedIn(c.shape):
                        click_pos = self.test_mouse.getPos()
                        click_time = core.monotonicClock.getTime()

                        # print(f"Mouse clicked on {c.global_idx}")
                        response_list.append(c.global_idx)
                        c_idxs.pop(idx)
                        clicked = True

                        # lowercase m monotonicClock (not MonotonicClock) starts counting
                        # from the time psychopy.core is imported
                        # add to logfile on each click
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
        return True

    def loop_experiment(self, candidates: List[StimulusImage], response_image_stim: StimulusImage, response_list: list,
                        click_data: list, candidate_tuple: Tuple[int], enforce_timeout: bool) -> bool:
        """

        Args:
            candidates:
            response_image_stim:
            response_list:
            click_data:
            candidate_tuple:
            enforce_timeout:

        Returns:
            A boolean value which indicates if the trial ran successfully, or if it timed out. True if success, False
            if timed out.
        """
        c_idxs = [idx for idx in range(len(candidates))]
        trial_start_time = time.time()
        while c_idxs:
            current_time = time.time()
            if enforce_timeout and (current_time - trial_start_time > self.trial_timeout):
                return False
            self.experiment_artist.draw_trial(response_image_stim, candidates, c_idxs)
            self.check_stim_clicked(candidates, response_list, click_data, candidate_tuple, c_idxs, trial_start_time,
                                    enforce_timeout)
        return True

    def oracle_runner(self, candidate_tuple: Tuple[int], enforce_timeout: bool) -> OracleResponse:
        """
        Runs the oracle and relevant graphic.
        """
        """
        print("\n\n================= NEW ROUND ====================")
        print(f"INDEX TO STIM: {self.global_idx_to_stim}")
        print(f"Working with tuple {candidate_tuple}")
        """
        response_list = [candidate_tuple[0]]
        response_image_stim = self.global_idx_to_stim[candidate_tuple[0]]
        candidates = [self.global_idx_to_stim[global_idx] for global_idx in candidate_tuple[1:]]
        # print(f"Candidates: {[c.global_idx for c in candidates]}")
        print(f"Enforce timeout?: {enforce_timeout}")

        click_data = []

        # ranking order from closest to farthest (0th position is closest to response)
        # draw the response
        self.experiment_artist.set_response_image_stim_params(response_image_stim)
        self.experiment_artist.set_candidate_image_stim_params(candidates)

        # run the experiment loop
        ran_successfully = self.loop_experiment(candidates, response_image_stim, response_list, click_data,
                                                candidate_tuple, enforce_timeout)

        self.num_tasks_done += 1

        self.experiment_artist.draw_loading_screen()
        if self.num_tasks_done > 29:
            core.wait(0.1)
        else:
            core.wait(0.5)

        if self.num_tasks_done % 90 == 0:
            self.experiment_artist.draw_and_wait_on_break_screen()
            self.experiment_artist.draw_bg_color([0, 0, 0])

        # print(f"responding with: {response_list}")
        return OracleResponse(tuple(response_list), click_data, not ran_successfully)

    def get_oracle(self):
        return self.oracle_runner
