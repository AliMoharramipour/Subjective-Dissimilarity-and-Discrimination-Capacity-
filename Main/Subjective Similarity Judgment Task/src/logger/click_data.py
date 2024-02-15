from dataclasses import dataclass
from typing import List


@dataclass
class ClickData:
    trial: int
    click_time: int
    click_pos: List[float]
    clicked_stim: int
    clicked_stim_pos: int
    candidate_stims: List[int]
    target_stim: int
