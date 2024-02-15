from dataclasses import dataclass
from typing import Tuple, List, Set

from experiment.stimulus_image import StimulusImage


@dataclass
class OracleResponse:
    response_tuple: Tuple[int]
    click_data: List[dict]
    timed_out: bool
