"""
- see design.txt for what needs to be implemented here. w2v_pipeline.ipynb has some minimal examples.
- take a look at raw_vocab.csv to see the tokens sorted by commonality (run importer.py first).

"""

import decimal as dec
from pathlib import Path
import pandas as pd
import nltk
import os
from datetime import datetime
import shutil
from tempfile import mkstemp
from gensim import utils
# todo: import sklearn train/test
from gensim.models.wrappers import FastText # poor memory usage
import fasttext
import random
import numpy as np
from util import downsample
import nltk.parse.stanford
from corpus import Corpus
import gensim.downloader as api
import time
# from corpus import Corpus
from sklearn.feature_extraction.text import TfidfVectorizer
from guppy import hpy; h = hpy()

DATA = Path('data')
TEMP = DATA / 'temp'
SEED = random.seed(0)

def lookup_embeddings(model_paths, dataset):

    ft_models = {}
    selection = ['PM']

    # assert that data is in the right shape
    try:
        assert dataset is type(pd.Series()) and dataset.name == 'Word'
    except AssertionError:
        try:
            dataset = dataset['Word']
        except:
            print('input to lookup_embeddings must be a doc_series')


    for key, path in model_paths.items(): # todo: cleanup & func. Avoid loading several models in a run.
        ft_models[key] = load_ft_model(key, path)

        # Generate word vector feature columns from loaded model
        selected_models = [(key, model) for key, model in ft_models.items() if key in selection]

        embeddings = {}
        for model_name, model in selected_models:

            featureset = pd.DataFrame(index=dataset.index)

        #     if features.columns.str.contains(model_name).any():
        #         print(f'Skipping feature generation: Features for \
        #                 {model_name} were already detected in the columns.')
        #         continue

            embeddings[model_name] = query_vectors(dataset, model, model_name)

            model_feats = pd.DataFrame(embeddings[model_name])

            model_feats.index = dataset.index
            model_feats.columns = [model_name + '_' + str(col) for col in model_feats.columns]

            features = pd.concat([dataset.loc[dataset.index], model_feats], axis=1)

            if len(features.columns) > 500:
                print(f'Warning: {len(features.columns)} columns passed.')

            featureset.to_pickle(DATA / 'features' / 'features_PM.pkl') # separate from other data


def load_ft_model(key, path):
    model = None

    if key[:2] == 'tr':
        model = FastText.load(str(path))  # For imports with gensim wrapper
    elif key[:2] == 'PM':
        try:
            print('Loading PubMed FastText model.')
            model = fasttext.load_model(str(path))  # for imports with base fastText package
            # models[key] = FastText.load_fasttext_format(str(path)) # causes memory issues

        except MemoryError:
            print(h.heap())

        print(f'Successfully loaded {key} model')


    return model


def query_vectors(dataset, model, model_name):
    model_type = model_name.split('_')[0]

    print('dataset just before vector query:\n')
    print(dataset.values)
    print(f'length of query input: {len(dataset.values)}')

    if model_type[:2] == 'tr':
        return [model[word] for word in dataset.values] # might not work anymore?

    elif model_type[:2] == 'PM':
        return [model.get_word_vector(word) for word in dataset.values]


def train_ft_models(datasets):

    for series in datasets:

        if not (DATA / 'ft_models' / ('tr_' + series.name + '.model')).exists():
            print(f'Learning embeddings for {series.name} dataset')
            learn_embedding(series, FastText)


def learn_embedding(series, model_f, **kwargs): # todo: move to depr if no longer used
    """
    Trains a FastText model based on a dataset. Only do training on the training dataset. Accepts keyword parameters
    for the modelling function.

    - both have parameters to tune.
    - will not explore if outperformed by pretrained methods.
    """

    docs = series.groupby('doc').apply(list).to_dict()

    model_path = DATA / 'ft_models' / ('tr_' + series.name + '.model')
    model_path.parent.mkdir(exist_ok=True)

    model = model_f(sentences=docs.values(), **kwargs) # todo: update with sent corp

    model.save(str(model_path))
    print(f'Model saved to {model_path}')

    return model # todo: temporary. Load models


def main():

    dataset = pd.read_pickle(DATA / 'raw' / 'labels.pkl')['Word'] # input to pipeline module; can pass thru sys.argv?

    corpus = Corpus(dataset) # todo: WIP
    # index = corpus.idx
    print('Imported raw data.')

    # corpus = corpus.idx_sentences()

    # corpus data handling here.


    if not (TEMP / 'base.pkl').exists(): # todo: remove later
        features = corpus.df
    else:
        features = pd.read_pickle(TEMP / 'base.pkl')

    if 'POS' not in features.columns:

        print('Generating NLTK POS tags')
        # print(corpus.idx_sentences().df)

        pos_feats = pd.DataFrame(index=features.index)
        pos_feats['POS'] = [tag for char, tag in nltk.pos_tag(corpus.df.values.flatten().tolist(),lang='eng')] # todo: -> corpus.py once testing complete // sanity check

        lags = [-2,-1,1,2]
        for lag in lags: # todo: wrap
            method = 'bfill' if lag > 0 else 'ffill'
            pos_feats[f'POS_LAG_{lag}'] = pos_feats['POS'].groupby('doc').shift(lag).fillna(method=method)

        pos_path = 'data/features/pos.pkl'
        pos_feats.to_pickle(pos_path)
        print(f'Saved NLTK POS tags and lag columns {lags} to {pos_path}')

        quit()

    raise NotImplementedError('FastText models are currently not available.')
    # todo: ---------------
    # todo: add TF/IDF col

    # todo: add more features

    #


    # todo: Stanford cols?

    # todo ---------------

    # add loaded models (start with fasttext pubmed)
    model_paths = {'PM': DATA / 'ft_models' / 'BioWordVec_PubMed_MIMICIII_d200.bin'
                   } # 'BioSentVec' for sentence level labeling?

    # debug downsample
    # dataset = downsample(dataset, frac=0.1)
    # index = dataset.index

    # todo: place these in if name main
    training_ft = False
    if training_ft:
        train_ft_models(model_paths, corpus.df) # currently untested.

    generate_embeddings = False
    if generate_embeddings:
        lookup_embeddings(model_paths, corpus.df) # don't need a return?

    # filename = 'piped_feats'
    # os.remove(TEMP / 'base.pkl') # todo: use tmpfile?
    # features.to_pickle(f'data/{filename}.pkl')

    # todo: rework data saving: save to compressed gzip.
    print('Finished generating features.')


if __name__ == '__main__':

    main()




