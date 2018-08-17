from framework import Context, ArgumentParser
from app import main, ArgumentParser


def before_execute(context: Context) -> None:
    context.logger.debug("Start processing...")


def execute(context: Context) -> None:
    args = ArgumentParser().parse()
    main.execute(context, args)


def after_execute(context: Context) -> None:
    context.logger.debug("Finished.")


if __name__ == "__main__":
    context = Context()
    before_execute(context)
    execute(context)
    after_execute(context)
