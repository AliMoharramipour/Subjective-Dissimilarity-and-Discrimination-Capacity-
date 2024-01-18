import time
from collections import defaultdict
from itertools import permutations, combinations

import numpy as np

from infotuple_py3 import body_metrics


class BodySelectorManager:
    def __init__(self, candidates: list, tuple_size: int,
                 body_selector=body_metrics.primal_body_selector_test, previous_selections=None):
        self.candidates = candidates
        self.tuple_size = tuple_size
        self.body_selector = body_selector
        if previous_selections is None:
            previous_selections = defaultdict(list)  # key is int
        self.previous_selections = previous_selections

    def run_body_selector_with_previous_selections(self, embedding: np.array, target_face_num: int,
                                                   body_selector_data_list):
        # TODO - We can store these permutations, like a backpack problem
        start_time = time.time()
        candidates = combinations([comparison_face_num for comparison_face_num in self.candidates
                                   if comparison_face_num is not target_face_num], self.tuple_size - 1)
        tuples = [[target_face_num] + list(comparison_face_num) for comparison_face_num in candidates]
        print(f"Time to calculate face combinations: {time.time() - start_time}")
        selected_tuple, tuple_qualities, _, _ = \
            self.body_selector(target_face_num, embedding, tuples, self.previous_selections, body_selector_data_list)

        self.previous_selections[target_face_num].append(selected_tuple)

        return selected_tuple, tuple_qualities
