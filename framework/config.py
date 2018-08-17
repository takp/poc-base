import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Config(object):
    def get(self, key: str) -> Optional[str]:
        return os.environ.get(key)
