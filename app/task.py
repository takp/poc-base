from framework import Context
import argparse

DATA_PATH = "data/"


class Task(object):
    def __init__(self):
        self.context = Context()
        self.args = self.get_arguments()

    def execute(self) -> None:
        """
        Add your code here.
        - Logging: self.context.logger.debug("logging output")
        - Env var: self.context.config.get("KEY")
        - Command Line Arguments: self.args.model_path
        """

    def get_arguments(self) -> argparse.Namespace:
        """
        Set your custom command line arguments here.
        (e.g.) parser.add_argument('--model', dest="model_path", help='set model path')
        """
        parser = argparse.ArgumentParser()
        return parser.parse_args()
