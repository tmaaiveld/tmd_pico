import random
import copy
import string
import warnings
from pathlib import Path
import pandas as pd

from processing.features.embeddings import load_ft_model
from processing.features.tfidf import tfidf
from processing.importer import load_tarfile
import processing.optimizer as opt

# nltk.download('stopwords')
CONLL_COLS = ['lemma', 'upos', 'xpos', 'feats', 'head', 'deprel']


class Corpus:
    """
    *** UPDATE DOCSTRING ***
    """

    def __init__(self, doc_series, indexed=False):

        # create two attributes for the documents and the features
        self.doc_series = doc_series.copy()
        self.df = doc_series.to_frame().rename({'Word': 'token'}, axis='columns')

        self.doc_ids = set(self.doc_series.index.unique('doc'))
        self.n_docs = len(self.doc_series.index.unique('doc'))

        # index the data
        if not indexed:
            self._idx_sentences()

    @classmethod
    def from_tarfile(cls, path, word_col='token'):
        return cls(load_tarfile(path)[word_col])

    @classmethod
    def from_parquet(cls, path, word_col='token'):
        df = pd.read_parquet(path)
        return cls.from_frame(df, word_col, indexed=False)

    @classmethod
    def from_frame(cls, df, word_col, indexed=True):

        corpus = cls(df[word_col], indexed=indexed)
        corpus.df = df
        return corpus

    def copy(self):
        return copy.deepcopy(self)

    def downsample(self, n=100):
        selected = random.sample(self.doc_ids, n)
        self.doc_series.index = self.df.index

        # sample_idx = pd.MultiIndex.from_frame(
        #     self.df[slice(None)].loc[(selected, slice(None), slice(None)), :]
        #         .reset_index(drop=True)#['doc', 'sent', 'word']
        # )

        sample_idx = self.df.index.to_frame().loc[(selected, slice(None), slice(None))].index#.drop('doc').set_index('doc', 'sent', 'word') \
                         #.loc[(selected, slice(None), slice(None)), :].index

        self.df = self.df.loc[sample_idx, :]
        self.doc_series = self.doc_series[sample_idx]
        self.doc_ids = selected
        self.n_docs = len(self.doc_ids)

    def process(self, *args):

        methods = []
        for i, arg in enumerate(args):
            methods += [name for name in dir(self) if arg in name]
        for method in methods:
            getattr(Corpus, method)(self)

        print(f'Added columns {methods}')

    def zip_sents(self):
        docs = self.doc_series
        docs.index = self.df.index
        return docs.groupby(['doc', 'sent']).apply(list).str.join(' ')

    def add_locators(self): # probably not needed
        # self.df['doc_loc'] = self.df.groupby('doc').cumcount() + 1
        self.df['sent_id'] = self.df.reset_index()['sent'].values
        self.df['sent_loc'] = self.df.reset_index()['word'].values
        return self

    def add_stem(self):
        import nltk
        stemmer = nltk.stem.snowball.SnowballStemmer(language='english')
        self.df['stem'] = [stemmer.stem(token) for token in list(self.doc_series)]
        return self

    def add_pos(self):
        from nltk import pos_tag
        # not used, included in CoNNL
        self.df['nltkpos'] = [tag for char, tag in pos_tag(self.doc_series.values.flatten().tolist(), lang='eng')]
        return self

    def add_pos_stem(self):
        if 'xpos' in self.df.columns:
            self.df['xpos_stem'] = (self.df['xpos']).str.slice(0, 2)
        elif 'nltkpos' in self.df.columns:
            self.df['nltkpos'] = (self.df['nltkpos']).str.slice(0, 2)
        else:
            raise Exception('No POS column found in data.')

    def add_tfidf(self):

        self.tfidf_ = tfidf(self.zip_sents().values.tolist())
        # print(self.tfidf_)
        # self.tfidf_tab = pd.DataFrame(self.tfidf_).iloc[:, 0].sort()

        # print(self.tfidf_tab)
        scores = self.doc_series.map(self.tfidf_).fillna(0.0)

        self.df['tfidf'] = scores
        self.df['sent_tfidf'] = scores.groupby(['doc', 'sent']).sum()
        return self

    def lag_cols(self, columns, window_size, level, fill_value='<NONE>'):
        lags = list(range(-window_size, window_size + 1))
        lags.remove(0)

        for column in columns:
            for lag in lags:
                self.df[f'{column.upper()}_LAG{lag}'] = self.df[column].groupby(level).shift(lag).fillna(fill_value)
        return self

    def mark_stopwords(self):
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english')) - {'no', 'not', 'nor'}
        self.df['stopword'] = self.doc_series.isin(stop_words).values
        return self

    def mark_punctuation(self):
        punctuation = set(string.punctuation)
        self.df['punctuation'] = self.doc_series.isin(punctuation).values
        return self

    def mark_capitals(self):
        self.df['is_upper'] = [token.isupper() for token in self.doc_series.values]
        self.df['is_lower'] = [token.islower() for token in self.doc_series.values]
        self.df['is_title'] = [token.istitle() for token in self.doc_series.values]

        if 'sent' in self.df.index.names:
            self.df['near_cap'] = self.df['is_upper'].groupby(['doc', 'sent']).sum() >= 1

        return self

    def mark_numeric(self):

        # print(self.doc_series.isna().sum())
        # print(self.doc_series[self.doc_series is None])

        self.df['is_int'] = self.doc_series.str.isdigit()
        self.df['is_dec'] = self.doc_series.apply(self.mark_decimal)
        return self

    @staticmethod
    def mark_decimal(token):
        import decimal
        try:
            decimal.Decimal(token)
            if token.isdigit(): raise decimal.InvalidOperation
            return True
        except decimal.InvalidOperation:
            return False
        except TypeError:
            print(token)

    def mark_first_last(self):

        print(self.df)
        if 'sent_id' not in self.df.columns:
            raise Exception('Add locator columns first.')

        n_sents_doc = self.df.groupby(['doc'])['sent_id'].transform(max)
        n_words_sent = self.df.groupby(['doc', 'sent'])['sent_loc'].transform(max)

        self.df['first_sent'] = self.df['sent_id'] == 1
        self.df['last_sent'] = self.df['sent_id'] == n_sents_doc
        self.df['first_word'] = self.df['sent_loc'] == 1
        self.df['last_word'] = self.df['sent_loc'] == n_words_sent
        return self

    def load_embeddings(self, file_path='data\\features\\ft_embeds.parquet',
                        model_path=None, pca_cutoff=0.9):

        if Path(file_path).exists():
            embeddings = pd.read_parquet(str(file_path))
            print(f'Loading FastText embeddings from {file_path}.')

        else:
            if model_path is not None:
                print(f'No embeddings found at {file_path}. Loading model for vector query. This may take a long time.')

                model = load_ft_model(model_path)
                embeddings = pd.DataFrame([model.get_word_vector(word) for word in self.doc_series.values],
                                          index=self.df.index, columns=[f'PMFT_{i + 1}' for i in range(200)])

                print('embeddings loaded. Applying Principal Component Analysis (set pca_cutoff to None to disable)')
                embeddings.to_parquet(str(file_path))
            else:
                raise Exception('Please specify a model_path to load the model from.')

        if not pca_cutoff is None:
            embeddings = opt.pca_transform(embeddings, pca_cutoff)

        self.df = self.df.join(embeddings)
        return self

    def load_CoNLL(self, file_path='data\\features\\conll.csv'):
        from processing.features.conll_parse import conll_parse

        if not Path(file_path).exists():
            print(f'No CoNNL output detected at {file_path}. Generating data. This may take up to several hours'
                  f'for the full dataset.')

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                conll_parse(self.zip_sents(), file_path)

        print(f'Loading CoNLL data from {file_path}.')

        index = self.df.index

        if str(file_path).split('.')[-1] == 'csv':
            output = pd.read_csv(file_path, sep='\t', header=None, keep_default_na=False).iloc[:, 2:8].dropna(axis=0, how='all')
        else:
            output = pd.read_parquet(file_path).iloc[:, 2:8].dropna(axis=0, how='all')

        output.columns = CONLL_COLS # todo: add cols

        # self.df = pd.concat([c.reset_index(drop=True) for c in [self.df, output]], axis=1)
        # self.df.index = index

        # output = pd.read_parquet(file_path).iloc[:, 2:8]

        nan = output.isna().sum().sum()
        if nan > 1:
            print(f"Warning: {nan} missing values detected in {file_path}. Filling with '_'.")
            output = output.fillna('_')

        self.df[CONLL_COLS] = output.fillna('_')

        return self

    def parse_deprel(self, parse_features):
        self.df['dist_to_parent'] = ~(self.df['head'] == 0) * abs(self.df['sent_loc'] - self.df['head'])

        head_orient = ~(self.df['head'] == 0) * (self.df['head'] - self.df['sent_loc'])
        head_loc = pd.Index((self.df.reset_index(drop=True).index + head_orient).reset_index(drop=True))

        parents = self.df.reset_index().loc[head_loc]
        parents[self.df.reset_index().index == head_loc] = '_'
        parents.columns = ['par_' + str(col) for col in parents.columns]

        self.df = pd.concat([self.df.reset_index(drop=True), parents[parse_features].reset_index(drop=True)], axis=1) \
            .set_index(self.df.index)

        # print(self.df)
        # print(parents)
        # print(self.df.index)
        # print(parents.index)
        #
        # self.df = self.df.join(parents)

        return self

    def load_sentiments(self, file_path='data\\processing\\sentiments.parquet'):

        if Path(file_path).exists():
            print(f'Loading sentiments from {file_path}')
            self.df[['polarity', 'subjectivity']] = pd.read_parquet(file_path)

        else:
            from textblob import TextBlob
            print(f'Running TextBlob sentiment analysis over {len(self.df)} instances.')

            self.df[['polarity', 'subjectivity']] = self.zip_sents() \
                .apply(lambda x: pd.Series(TextBlob(x).sentiment))

        return self

    def save(self, path):
        ext = str(path).split('.')[-1]
        if ext == 'parquet':
            self.df.to_parquet(path)
        elif ext == 'pickle':
            self.df.to_pickle(path)
        elif ext == 'csv':
            self.df.to_csv(path)
        else:
            print('Invalid file extension specified. Supported formats: .pickle, .parquet, .csv (save only)')

    def load_df(self, path):
        ext = str(path).split('.')[-1]
        if ext == 'parquet':
            self.df = self.df.read_parquet(path)
        elif ext == 'pickle':
            self.df = self.df.read_pickle(path)
        else:
            print('Invalid file extension specified. Supported formats: .pickle, .parquet')


    def _idx_sentences(self):
        print(f'Indexing sentences for {self.n_docs} documents.')
        new_indices = self.doc_series.copy().to_frame()

        sent_break = new_indices.mask(new_indices == '.', 1).mask(new_indices != '.', 0)

        sent_id = sent_break.groupby('doc').shift(1).fillna(method='bfill') \
                            .groupby('doc').expanding().sum().astype(int).values

        new_indices['sent_idx'] = sent_id + 1

        new_indices['word_idx'] = (new_indices.set_index('sent_idx',append=True)
                                              .groupby(['doc', 'sent_idx']).cumcount() + 1).values

        self.df.index = pd.MultiIndex.from_frame(
            new_indices.reset_index().drop('idx', axis=1)[['doc', 'sent_idx', 'word_idx']],
            names=['doc', 'sent', 'word']
        )

        self.doc_series.index = self.df.index

        def _optimize_for_tree(self):
            self.df = opt.tree_optimize(self.df)
            return self


if __name__ == '__main__':
    testfile = 'path/to/testfile.parquet'
    data = pd.read_parquet(testfile)['form']
    corpus = Corpus(data)

    corpus.mark_numeric()
    corpus.mark_capitals()
    corpus.load_sentiments()

    print(corpus.df)
    print(corpus.df.columns)
    print(corpus.df.index)
    print(corpus.doc_series)
    print(corpus.df.sum())
