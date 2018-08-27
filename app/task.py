from framework import BaseTask

DATA_PATH = "data/"


class Task(BaseTask):
    """
    Concrete task class.
    """

    def __init__(self):
        super().__init__()

    def execute(self) -> None:
        """
        Concrete execute method.

        Notes
        -----
        1. Logging:
            You can output logs with `self.context.logger`.
            (e.g.) self.context.logger.debug("logging output")
        2. Env var:
            You can access to environment variables with `self.context.config`.
            (e.g.) self.context.config.get("KEY")
        3. Command Line Arguments:
            You can access to arguments through `self.args` after set your arguments
            through `set_arguments` method.
            (e.g.) self.args.model_path
        """

    def set_arguments(self, parser) -> None:
        """
        Set your command line arguments if necessary.

        Notes
        -----
        Adding command line arguments.
        (e.g.) `parser.add_argument('--model', dest="model_path", help='set model path')`
        """
