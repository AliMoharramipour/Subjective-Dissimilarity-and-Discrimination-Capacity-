from stair_handler_get_intensity import StairHandlerGetIntensity
from staircase_logger import StaircaseLogger


class StairHandlerWithCustomLogging:
    def __init__(self, stair: StairHandlerGetIntensity):
        self.stair = stair
        self.timed_out = False

        self.filename = f"staircase_{self.stair.extraInfo['face_pair']}.log"
        self.logger = StaircaseLogger(self.filename)

    def write_stair_as_pickle(self, filename=""):
        if filename == "":
            filename = self.filename
        self.stair.saveAsPickle(filename, fileCollisionMethod="overwrite")
