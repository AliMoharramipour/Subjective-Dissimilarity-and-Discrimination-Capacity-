import time
from collections import defaultdict
from itertools import combinations

import numpy as np
import random
import copy

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

    def run_body_selector_with_previous_selections(self, embedding: np.array, first_candidate: int,
                                                   body_selector_data_list, fixed_candidates=tuple()):
        # TODO - We can store these permutations, like a backpack problem
        candidate_use = copy.deepcopy(self.candidates)
        random.shuffle(candidate_use)

        start_time = time.time()
        if len(fixed_candidates) == 0:
            candidates = combinations([comparison_face_num for comparison_face_num in candidate_use
                                    if comparison_face_num is not first_candidate],
                                    self.tuple_size - 1)
            fixed_faces = [first_candidate]
            tuples = [fixed_faces + list(comparison_face_num) for comparison_face_num in candidates]
        else:
            candidates = combinations([comparison_face_num for comparison_face_num in candidate_use
                                    if (comparison_face_num is not first_candidate) and (comparison_face_num is not fixed_candidates[0])],
                                    self.tuple_size - 1 - len(fixed_candidates))
            fixed_faces = [first_candidate]
            tuples = []
            for comparison_face_num in candidates:
                selected_candidate_here = list(comparison_face_num) + fixed_candidates
                random.shuffle(selected_candidate_here)
                tuples.append(fixed_faces + selected_candidate_here)

        print(f"Time to calculate face combinations: {time.time() - start_time}")
        selected_tuple, tuple_qualities, _, _ = \
            self.body_selector(first_candidate, embedding, tuples, self.previous_selections, body_selector_data_list)

        self.previous_selections[first_candidate].append(selected_tuple)

        return selected_tuple, tuple_qualities
