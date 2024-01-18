from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class OracleResponse:
    response_tuple: Tuple[int]
    click_data: List[dict]
