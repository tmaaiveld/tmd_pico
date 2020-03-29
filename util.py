'''
todo:
- check what is still used / needed in this file.
'''

import json
import random
from itertools import chain
from pathlib import Path
import pandas as pd
import sys


def c(x):
    return chain.from_iterable(x)


def get_id(path):
    return int(''.join(x for x in path if x.isdigit()))


def make_dirs(*args):
    [arg.mkdir(parents=True, exist_ok=True) for arg in args]


def get_wv(word, wv):
    return wv[word]


def split_docs(token_s):
    return token_s.groupby('doc').apply(list).tolist()


def downsample(df, frac):
    n = int(len(df.index.unique('doc')) * frac)
    print(n)
    print(df.index)
    selected = random.sample(list(df.index.unique('doc')), n)
    return df.loc[(selected, slice(None))]


def mem_usage(*args):
    data_mem = sum(sys.getsizeof(i) for i in [args])
    print(f'Loaded {data_mem / (10**9)} GB of data')


if __name__ == '__main__':
    make_dirs('data/raw')

