# import random
#
# import numpy as np
# import pandas as pd
# from pathlib import Path
#
# from corpus import Corpus
#
# DATA_PATH = Path('data\\raw\\labels.parquet')
# SEED = random.seed(0)
# DOWNSAMPLE = False
#
# dataset = pd.read_parquet(DATA_PATH)['Word']
# doc_ids = dataset.index.get_level_values('doc')
#
# # if DOWNSAMPLE:
# #     n_docs = 25
# #     sample_ids = random.sample(set(doc_ids), n_docs)
# #     data_ds = dataset[doc_ids.isin(sample_ids)].copy()
# #
# #     dataset = data_ds
#
# corpus = Corpus(dataset)
#
# corpus.add_locators() # add index position information
# # corpus.process_tokens(None) # CoNNL already adds a 'form'
# corpus.process_tokens('stem') # add the stem as a feature
#
# # add binary markers for attributes
# corpus.mark_stopwords()
# corpus.mark_punctuation()
# corpus.mark_capitals()
# corpus.mark_numeric()
# corpus.mark_first_last()
#
# corpus.load_embeddings()
#
# corpus.load_CoNLL(filepath='data/conll.csv')
#
# parental_features = ['par_' + col for col in ['form', 'lemma', 'deprel', 'upos', 'xpos']]
# corpus.parse_deprel(parental_features) # parse the parental relations in the CoNNL rows
#
# corpus.calc_tfidf()
#
# assert 'xpos' in corpus.df.columns
# corpus.add_pos_stem() # add a stemmed version of the CoNNL XPOS (basic grammar info, i.e. NN, VB)
#
# # lags
# window = 2
# lag_features = ['is_int', 'is_dec', 'first_word', 'last_word', 'form', 'lemma', 'upos', 'xpos', 'deprel', 'tfidf']
# par_cols = corpus.df.columns[corpus.df.columns.str.startswith('par')]
# fill_tag = '_' # fill missing strings with a custom value
#
# corpus.add_lag(['is_int', 'is_dec', 'first_word', 'last_word'], window, level='doc', fill_value=False)
# corpus.add_lag(['form', 'lemma', 'upos', 'xpos', 'deprel'], window, level='doc', fill_value=fill_tag)
# corpus.add_lag(['tfidf'], window, level='doc', fill_value=0)
#
# corpus.df[par_cols] = corpus.df[par_cols].fillna(fill_tag)
#
# print(corpus.df)
# print(len(corpus.df))
# print(corpus.df.isna().sum())
# print(corpus.df.isna().sum(axis=1))
# assert corpus.df.isna().sum().sum() == 0
#
# print('processing in the final set: \n', list(corpus.df.columns))
#
# corpus.df.head(30)
#
# corpus.save('data\\processing\\dataset.parquet')