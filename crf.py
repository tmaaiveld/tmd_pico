import random
import pickle
import time
import warnings
import itertools
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn_crfsuite
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn_crfsuite import metrics

pd.options.display.float_format = '{:20,.5f}'.format

SEED = random.seed(0)
EXPERIMENTS = Path('experiments')
dir_name = 'CRF_allfeats'
PIO = {"participants", "interventions", "outcomes"}

def aggregate_labels(lab_df, level='sent'):
    """Can aggregate the labels if desired. Currently not (properly) implemented."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        lab_df['none'] = (lab_df.sum(axis=1) == 0).astype(int)
        lab_df['label'] = ['none'] * len(lab_df)

        for label in list(PIO):
            lab_df.loc[lab_df[label] == 1, 'aggregate'] = label

    return lab_df


def dictify(docs_df):
    x = docs_df.groupby(level=['doc', 'sent']).apply(
        lambda df: list(df.xs(df.name).reset_index(drop=True).T.to_dict().values())
    )

    return list(x)


train_val = pd.read_parquet('data\\split\\train.parquet')

get = lambda pat, s: {v for v in s if pat in v}

labels = PIO.union({'aggregate'})
features = {}

features['all'] = set(train_val.columns.difference(PIO))
features['fasttext'] = get('PMFT', features['all'])
features['lag'] = get('LAG', features['all'])
features['deprel'] = {'deprel', 'dist_to_parent', 'par_form', 'par_lemma', 'par_upos', 'par_xpos'}
del get

k_folds = 5

train_idx, val_idx = train_test_split(train_val.index.unique('doc'),
                                      train_size=1-(1/k_folds)) # default 80:20

print(f'splitting data {len(train_idx)}:{len(val_idx)}')

train_set = train_val.loc[(train_idx, slice(None)),:]
val_set   = train_val.loc[(val_idx, slice(None)),:]

X_train_df, y_train_df = train_set[features['all']], train_set[PIO]
X_test_df, y_test_df = val_set[features['all']], val_set[PIO]

y_train_df, y_test_df = [aggregate_labels(df) for df in [y_train_df, y_test_df]]

X_train, y_train, X_test, y_test = [dictify(df) for df in [X_train_df, y_train_df, X_test_df, y_test_df]]

f"Finished formatting the data"

def aggregate_labels(lab_df, level='sent'):
    """Can aggregate the labels if desired. Currently not (properly) implemented."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        lab_df['none'] = (lab_df.sum(axis=1) == 0).astype(int)
        lab_df['label'] = ['none'] * len(lab_df)

        for label in list(PIO):
            lab_df.loc[lab_df[label] == 1, 'aggregate'] = label

    return lab_df


def dictify(docs_df):
    x = docs_df.groupby(level=['doc', 'sent']).apply(
        lambda df: list(df.xs(df.name).reset_index(drop=True).T.to_dict().values())
    )

    return list(x)

print('done')