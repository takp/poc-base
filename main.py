from framework import settings, Logger


def execute():
    logger = Logger()
    password = settings.get("SAMPLE_PASSWORD")
    logger.debug("Password: {}".format(password))
    logger.info("Completed")


if __name__ == "__main__":
    execute()
