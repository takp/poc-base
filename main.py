from framework import Context
from app import main


def before_execute(context: Context):
    context.logger.debug("Start processing...")


def execute(context: Context):
    """Add you code here."""
    main.execute(context)


def after_execute(context: Context):
    context.logger.debug("Finished.")


if __name__ == "__main__":
    context = Context()
    before_execute(context)
    execute(context)
    after_execute(context)
