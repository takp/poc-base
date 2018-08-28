import os

from framework import BaseTask

from .text_classify  import predict

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

        self.context.logger.debug("Start: sample1")

        data_path = os.path.join(os.path.dirname(__file__), "./text_classify/data/tweets_clean_2k.csv")
        result_path = os.path.join(os.path.dirname(__file__), "./text_classify/data/tweets_clean_2k_out.csv")

        # data_path = os.path.join(os.path.dirname(__file__), "./text_classify/data/tweets_clean.csv")
        # result_path = os.path.join(os.path.dirname(__file__), "./text_classify/data/tweets_clean_out.csv")

        dataset = predict.processFile(data_path, result_path)
        _ = predict.checkAccuracy(dataset, True)

        self.context.logger.debug("End: sample1")

        #
        # print(self.context.config.get("SAMPLE_PASSWORD"))
        #
        # print(self.args.model_path)


    def set_arguments(self, parser) -> None:
        """
        Set your command line arguments if necessary.

        Notes
        -----
        Adding command line arguments.
        (e.g.) `parser.add_argument('--model', dest="model_path", help='set model path')`
        """
        parser.add_argument('--model', dest="model_path", help='set model path')