from datetime import datetime

from psychopy.data import StairHandler
from staircase_logger import StaircaseLogger


class StairHandlerWithCustomLogging:
    def __init__(self, stair: StairHandler):
        self.stair = stair

        self.filename = f"staircase_{self.stair.extraInfo['face_pair']}.log"
        self.logger = StaircaseLogger(self.filename)

    def write_stair_as_pickle(self, filename=""):
        if filename == "":
            filename = self.filename
        self.stair.saveAsPickle(filename)
