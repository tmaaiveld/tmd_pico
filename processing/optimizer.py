import random
import pickle
import time
import warnings
import itertools
from datetime import datetime
from pathlib import Path
import xgboost

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


def mem_usage(pandas_obj):
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:  # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2  # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)


def convert_strings(df):
    converted_df = pd.DataFrame()
    for col in df.columns:
        num_unique_values = len(df[col].unique())
        num_total_values = len(df[col])
        if num_unique_values / num_total_values < 0.5:
            converted_df.loc[:, col] = df[col].astype('category')
        else:
            converted_df.loc[:, col] = df[col]
    return converted_df


def get_top_words(df, col, n=1000):
    return df.drop(df.index[df[col].duplicated()]) \
               .sort_values('tfidf', ascending=False).iloc[:n][col]


# def hotcode(df):
#     num_cols = df._get_numeric_data().columns
#     cat_cols = (set(df.columns) - set(num_cols)) - {'Word'}
#
#     print(cat_cols)
#     dummies = pd.get_dummies(df[cat_cols], sparse=True, columns=cat_cols)
#
#     print('hotcode complete.')
#     # assert check that type is numeric for all
#
#     return pd.concat([dummies, df[num_cols]], axis=1)


df = pd.read_parquet('data\\dataset.parquet')


print(list(df.columns))


start_time = time.time()

top_lemmas = set(get_top_words(df, 'lemma'))
replace_lemmas = {k: '_' for k in set(df['lemma']) - top_lemmas}

print('replacing lemmas')

for col in ['lemma', 'par_lemma']:
    df[col] = df[col].where(~(df['is_int']), 'xNUMx')
    df[col] = df[col].where(~(df['is_dec']), 'xDECx')
    print(df[col])
    df[col] = df[col].map(replace_lemmas).fillna(df[col])
    df[col] = df[col].fillna(df['token'])
df = df.fillna('_')

# processing = set(df.columns)
drop_feats = {'token', 'feats', 'stem', 'head', 'par_form', 'CoNNL_ID',
              'IS_INT_LAG-2', 'IS_INT_LAG-1', 'IS_INT_LAG1', 'IS_INT_LAG2', 'IS_DEC_LAG-2',
              'IS_DEC_LAG-1', 'IS_DEC_LAG1', 'IS_DEC_LAG2', 'FIRST_WORD_LAG-1', 'FIRST_WORD_LAG-2',
              'LAST_WORD_LAG2', 'LAST_WORD_LAG1', 'LEMMA_LAG-2', 'LEMMA_LAG-1', 'LEMMA_LAG1', 'LEMMA_LAG2',
              'XPOS_LAG-2', 'XPOS_LAG-1', 'XPOS_LAG1', 'XPOS_LAG2', 'DEPREL_LAG-2', 'DEPREL_LAG-1',
              'DEPREL_LAG1', 'DEPREL_LAG2', 'FORM_LAG-2', 'FORM_LAG-1', 'FORM_LAG1', 'FORM_LAG2', 'form', 'par_form'}

to_drop = set(df.columns).intersection(drop_feats)

df = df.drop(to_drop, axis=1)
print('converting dtypes')
int_dc = df.select_dtypes(include=['int32']).copy().apply(pd.to_numeric,downcast='signed') \
                  .join(df.select_dtypes(include=['int64']).copy().apply(pd.to_numeric,downcast='signed'))
float_dc = df.select_dtypes(include=['float64']).apply(pd.to_numeric,downcast='float')
str_dc = convert_strings(df.select_dtypes(include=['object']).copy())

print('dropping cols')
drop_cols = list(int_dc.columns) + list(float_dc.columns) + list(str_dc.columns)
print([df.drop(drop_cols, axis=1), int_dc, float_dc, str_dc])

df = pd.concat([df.drop(drop_cols, axis=1), int_dc, float_dc, str_dc], axis=1)

unsigned = df.select_dtypes(include=['uint8', 'uint16']).columns
for col in unsigned:
    df[col] = df[col].apply(pd.to_numeric, downcast='signed')

print(time.time()-start_time)

df.to_parquet('data\\dataset3.parquet')


# get = lambda pat, s: {v for v in s if pat in v}
#
# labels = PIO.union({'aggregate'})
#
# processing = {}
#
# processing['all'] = set(train_val.columns.difference(PIO))
# processing['fasttext'] = get('PMFT', processing['all'])
# processing['lag'] = get('LAG', processing['all'])
# processing['deprel'] = {'deprel', 'dist_to_parent', 'par_form', 'par_lemma', 'par_upos', 'par_xpos'}
#
# k_folds = 5
# train_idx, val_idx = train_test_split(train_val.index.unique('doc'),
#                                       train_size=1-(1/k_folds)) # default 80:20
#
# print(f'splitting data {len(train_idx)}:{len(val_idx)}')
#
# train_set = train_val.loc[(train_idx, slice(None)),:]
# val_set   = train_val.loc[(val_idx, slice(None)),:]
#
# X_train_df, y_train_df = train_set[processing['all']], train_set[PIO]
# X_test_df, y_test_df = val_set[processing['all']], val_set[PIO]
#
# print('hotcoding categorical columns...')
# try:
#     train_val = hotcode(train_val)
# except ValueError:
#     print('No categorical values found in data')