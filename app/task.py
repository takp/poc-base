from framework import Context
from .argument_parser import ArgumentParser

DATA_PATH = "data/"


class Task(object):
    def __init__(self):
        self.context = Context()
        self.args = ArgumentParser().parse()

    def execute(self) -> None:
        """
        Add your code here.
        - Logging: self.context.logger.debug("logging output")
        - Env var: self.context.config.get("KEY")
        - Command Line Arguments: You can access through "args". (foo = args["foo"])
        """
