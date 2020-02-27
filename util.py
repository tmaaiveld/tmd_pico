import json
from itertools import chain

def c(x):
    return chain.from_iterable(x)

def load_json(path):
    with path.open() as f:
        return json.load(f)

def get_id(path):
    return int(''.join(x for x in path if x.isdigit()))