from collections import defaultdict
from typing import List


class SimTracker:
    def __init__(self):
        self.sim_tracker = defaultdict(list)

    def add_oracle_result(self, oracle_sorted_tuple: tuple):
        target = oracle_sorted_tuple[0]
        candidates = oracle_sorted_tuple[1:]
        for c1 in range(len(candidates)):
            for c2 in range(c1 + 1, len(candidates)):
                self.sim_tracker[target].append((candidates[c1], candidates[c2]))

    def add_oracle_result_list(self, oracle_sorted_tuples: List[tuple]):
        for oracle_sorted_tuple in oracle_sorted_tuples:
            self.add_oracle_result(oracle_sorted_tuple)
