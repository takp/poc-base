import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def get(key: str) -> Optional[str]:
    return os.environ.get(key)
