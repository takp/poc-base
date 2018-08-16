import logging
import time


class Logger():
    def __init__(self):
        self.start_time = time.time()
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    def warning(self, msg):
        logging.warning(self.add_time(msg))

    def info(self, msg):
        logging.info(self.add_time(msg))

    def debug(self, msg):
        logging.debug(self.add_time(msg))

    def add_time(self, msg):
        time_spent = round((time.time() - self.start_time), 3)
        return "[{}] {}".format(time_spent, msg)
