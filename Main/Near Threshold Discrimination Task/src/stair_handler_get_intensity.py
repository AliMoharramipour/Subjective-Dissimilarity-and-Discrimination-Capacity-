from psychopy.data import StairHandler


class StairHandlerGetIntensity(StairHandler):
    def get_intensity(self):
        """The intensity (level) of the current staircase"""
        return self._nextIntensity

    def __str__(self):
        return (f"Name: {self.name}, "
                f"Trial #: {self.thisTrialN}, "
                f"Current intensity: {self.get_intensity()}")
