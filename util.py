import json
from itertools import chain
from pathlib import Path


def c(x):
    return chain.from_iterable(x)


def load_json(path):
    with path.open() as f:
        return json.load(f)


def get_id(path):
    return int(''.join(x for x in path if x.isdigit()))


def create_folders(*args):
    [Path(arg).mkdir(parents=True, exist_ok=True) for arg in args]
