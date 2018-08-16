from framework import settings


def execute():
    """Add your python code here"""
    password = settings.get("SAMPLE_PASSWORD")
    print(password)
