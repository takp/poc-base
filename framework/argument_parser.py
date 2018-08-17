import argparse


class ArgumentParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def add(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def parse(self):
        return self.parser.parse_args()
