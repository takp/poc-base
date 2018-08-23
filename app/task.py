from framework import BaseTask

DATA_PATH = "data/"


class Task(BaseTask):
    def __init__(self):
        super().__init__()

    def execute(self) -> None:
        """
        Add your code here.
        - Logging: self.context.logger.debug("logging output")
        - Env var: self.context.config.get("KEY")
        - Command Line Arguments: You can access to arguments through self.args.
          (e.g.) self.args.model_path
        """

    def set_arguments(self, parser) -> None:
        """
        If you want to add command line arguments, set arguments here.
        (e.g.) parser.add_argument('--model', dest="model_path", help='set model path')
        """
