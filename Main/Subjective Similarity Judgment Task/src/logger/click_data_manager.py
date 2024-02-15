import csv
import dataclasses

from logger.click_data import ClickData


class ClickDataManager:
    """Manages click data.
    Not multithread/multiprocess safe.
    """
    index = 0

    def __init__(self, is_enabled: bool, output_dir: str):
        self.is_enabled = is_enabled
        if is_enabled:
            self.logfile = open(f"{output_dir}/click_data.csv", "x", newline="")
            fieldnames = [field.name for field in dataclasses.fields(ClickData)]
            fieldnames.insert(0, "")  # support index column
            self.writer = csv.DictWriter(
                self.logfile,
                fieldnames=fieldnames,
                delimiter=",")
            self.writer.writeheader()

    def record_click(self, click_data: ClickData):
        if self.is_enabled:
            this_click = dataclasses.asdict(click_data)
            this_click[""] = self.index
            self.writer.writerow(this_click)
            self.index += 1

    def close_logfile(self):
        self.logfile.close()
