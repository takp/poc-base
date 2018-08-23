from framework import Context
import argparse

DATA_PATH = "data/"


class BaseTask(object):
    """
    Don't edit here. Add your code to `app/task.py`.
    """

    def __init__(self):
        self.context = Context()
        self.args = self.get_arguments()

    def execute(self) -> None:
        pass

    def get_arguments(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        self.set_arguments(parser)
        return parser.parse_args()

    def set_arguments(self, parser) -> None:
        pass
