import typing

from collections import defaultdict

from psychopy.data import StairHandler
from psychopy.tools.filetools import openOutputFile
from typing import List


class StaircaseLogger:
    reversals = []
    responses = defaultdict(list)

    def __init__(self, filename: str):
        self.filename = filename

    def store_response(self, stair_name: str, morph_id: int, distance: int, response: bool,
                       morph_ids: List[int]) -> None:
        self.responses[stair_name].append({"morph_id": morph_id, "distance": distance, "response": response,
                               "morph_ids": morph_ids})

    @staticmethod
    def write_reversals(f: typing.TextIO, stair: StairHandler) -> None:
        """
        Janky because it uses PsychoPy stair's built-in reversal list.
        :param f: File handle to write to
        :param stair: The staircase which has the reversal information
        :return: None
        """
        f.write("reversalDistances")
        for intensity in stair.reversalIntensities:
            f.write(f"\t{intensity}")
        f.write("\n")
        f.write("reversalIndices")
        for index in stair.reversalPoints:
            f.write(f"\t{index}")  # this is the trial number at the reversal
        f.write("\n")

    def write_responses(self, f: typing.TextIO, stair: StairHandler) -> None:
        print(stair.name)
        responses = self.responses[stair.name]
        print(self.responses)
        f.write("responseDistances")
        for info in responses:
            f.write(f"\t{info['distance']}")
        f.write("\n")
        f.write("responseValues")
        for info in responses:
            f.write(f"\t{info['response']}")
        f.write("\n")
        f.write("responseMorphIDs")
        for info in responses:
            f.write(f"\t{info['morph_id']}")
        f.write("\n")
        f.write("candidateMorphs")
        for info in responses:
            f.write(f"\t{info['morph_ids']}")
        f.write("\n")

    def write_logfile(self, stair: StairHandler, filename="", output_dir=".") -> None:
        if filename == "":
            filename = self.filename

        f = openOutputFile(f"{output_dir}/{filename}", append=False, fileCollisionMethod="overwrite",
                           encoding="utf-8-sig")
        self.write_reversals(f, stair)
        self.write_responses(f, stair)

        f.close()
