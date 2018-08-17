import argparse


class ArgumentParser(object):
    def parse(self) -> dict:
        """
        Add your custom command line arguments here.
        The argument value will be passed to "app/main.py" through "args".
        (e.g.)
        parser = argparse.ArgumentParser()
        parser.add('--foo', dest="foo", help='set your foo', default="123")
        return parser.parse_args().__dict__
        """
        parser = argparse.ArgumentParser()
        return parser.parse_args().__dict__
