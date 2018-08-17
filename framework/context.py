from .config import Config
from .logger import Logger


class Context(object):
    def __init__(self):
        self.logger = Logger()
        self.config = Config()
