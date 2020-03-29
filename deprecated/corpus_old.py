import pandas as pd
import decimal as dec
from nltk.corpus import stopwords
import numpy as np
import nltk
#nltk.download('stopwords')

class Corpus: # move later
    """
    A class which receives a pd.Series of token strings indexed by document. Attributes include a doc_series
    which retains the unchanged time series text data, a df to which new features may be appended. The object can be
    loaded in any module by passing a loaded document series and d can execute various preprocessing methods to
    produce different version of word columns or different features to be added.
    """

    # todo: add a method for detecting stop words


    # https://radimrehurek.com/gensim/parsing/preprocessing.html
    # https://www.nltk.org/_modules/nltk/parse/stanford.html

    def __init__(self, doc_series):
        self.doc_series = doc_series # remains unchanged
        # self.idx = doc_series.index
        self.df = doc_series.to_frame() # is changed by corpus methods
        self.doc_lists = doc_series.groupby('doc').apply(list).tolist()

        self.sent_lists = None
        self.sent_index = None
        self.idx_sentences()

    def remove_whitespace(self):
        self.df['Word'] = [word.strip() for word in self.df['Word']]
        return self

    def lower(self):
        self.df['Word'] = [word.lower() for word in self.df['Word']]
        return self

    def stem(self): # todo: untested
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        self.df['Word'] = [stemmer.stem(word) for word in self.df['Word']]
        return self

   # def lemmatize(self):
   #     lemmatizer = WordNetLemmatizer()
   #     self.df['Word'] = [lemmatizer.lemmatize(word, pos=postag)for word in self.df['Word']]

    def replace_num(self):
        """
        Currently doesn't work.
        """
        raise NotImplementedError('handle_numeric is not working')
        # .apply applies a function to every item of the dataframe.
        self.df['Word'] = self.df['Word'].apply(self.handle_numeric).values

        return self

    # @staticmethod tells Python a method is only placed in the class for readability purposes.
    @staticmethod
    def handle_numeric(lemma):
        if lemma.isnumeric():
            return '<NUM>'
        else:
            try:
                dec.Decimal(lemma)
                return '<DEC>'
            except dec.InvalidOperation:
                return lemma

    def idx_sentences(self):
        # First, slice the word column from the Corpus.df. .to_frame() converts that column to a new DataFrame.
        new_indices = self.df['Word'].copy().to_frame()

        # this code calculates the sentence ID number for each token and adds it to the new_indices DataFrame.
        sent_break = new_indices.mask(new_indices == '.', 1)                     \
                                .mask(new_indices != '.', 0)                     \
                                .droplevel(level='idx')                          \
                                .fillna(method='bfill')

        sent_id = sent_break.groupby('doc').shift(1).fillna(method='bfill')      \
                            .groupby('doc').expanding().sum().astype(int).reset_index(drop=True)

        new_indices['sent'] = sent_id.values + 1
        new_indices['token'] = 1
        new_indices = new_indices.set_index('sent', append=True).reorder_levels(['doc','sent','idx'])

        new_indices['token'] = new_indices['token'].droplevel(level='idx')                      \
                                                   .groupby(['doc', 'sent']).expanding().sum()  \
                                                   .astype(int).values

        new_indices = new_indices.reset_index('sent').drop('Word', axis=1)

        # The resulting index is saved separately.
        self.sent_index = new_indices.set_index(['sent', 'token'], append=True) \
                                     .reset_index('idx', drop=True).index

        # Last, we return self to allow this method to be chained with other methods of the Corpus object.
        return self

    def by_sent(self):
        self.df.index = self.sent_index
        return self

    def loc_features(self):
        new_indices = pd.DataFrame(index=self.sent_index).reset_index(['doc', 'sent', 'token']).astype(str)
        self.df['doc_loc'] = new_indices['doc'].values
        self.df['sent_loc']  = new_indices['sent'].values
        self.df['token_loc'] = new_indices['token'].values
        return self

    def first_last_features(self):
        new_col = len(self.df.index) * [0]
        self.df['first_sent'] = new_col
        self.df['last_sent'] = new_col
        print(self.df)
        documents = self.df['doc_loc'].tolist()
        # for doc in documents:
            #selection['first_sent'] = [1 for sent_loc in selection['sent_loc'] if sent_loc == 1]
            #selection['last_sent'] = [1 for sent_loc in selection['sent_loc'] if sent_loc == n_last_sent]

        # selection = self.df.loc[(self.df['doc_loc'] == doc)]
        # n_last_sent = len(set(selection['sent_loc'].tolist()))
        last_sents = self.df['sent_loc'].groupby('doc').max().to_string()[1000:]

        self.df = self.df.merge(last_sents, method='left', on='doc')
        # self.df.loc['sent_loc'] = last_sents

        print(self.df['sent_loc'])
        quit()
        #.groupby('doc').transform('max')
        # self.df[self.df.groupby('doc')['last_sent'].transform('max') == df['sent_loc']]

        print(self.df)
        print(last_sents)
        quit()
        x = selection.where(selection['sent_loc' == 1])

        print(x)
        quit()

            # for i, r in selection.iterrows():
            #     if r['sent_loc'] is 1:
            #         r['first_sent'] = 1
            #     elif r['sent_loc'] is n_last_sent:
            #         r['last_sent'] = 1
        print(self.df)
        return self

    def stopword_feature(self):

        new_col = len(self.df.index) * [0]
        self.df['stopword'] = new_col

        self.df['newcol'] = self.df['Word'].apply(self.find_stopword)
        # for index, row in self.df.iterrows():
        #     if row['Word'] in stop_words:
        #         row['stopword'] = 1

        print(self.df['stopword'][1])

        #select_indices = list(np.where(self.df["Word"] == True)[0])
        #df.iloc[select_indices]
        #for token in self.df['Word']:
        #    if token in stop_words:
        #        self.df.loc[[token], ['stopword']] = 1
                #print(self.df.loc[token])
                #self.df.loc[token]['stopword'] = 1   #or replace with a nonsense token that has no influence in training?
        print(self.df)
        return self

    @staticmethod
    def find_stopword(string):
        x = {'not', 'no'}
        stop_words = set(stopwords.words('english')) - x

        if string in stop_words:
            return 1
        return 0


    def zip_sents(self):
        # check if data is correctly formatted?
        docs = self.doc_series
        docs.index = self.sent_index
        return docs.groupby(['doc', 'sent']).apply(list).to_dict()



if __name__ == '__main__':

    # dataset production demo
    # could use parent/child (DocSeries parent, Corpus child)
    print('loading data')
    data = pd.read_parquet('data/split/train_1000.parquet')['Word']
    corp = Corpus(data)
    corp.by_sent() # index the corpus by sentence

    # zip the sentences to a dict of lists for parsing and save them to a list using pickle
    zip_sents = corp.zip_sents()
    #import pickle
    #with open('data/sentences.pickle', 'wb') as handle:
    #    pickle.dump(zip_sents, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # add word/sent location features to df

    corp.loc_features()    #adds sent id and token id as features to df
    corp.first_last_features()  #adds whether token is in first or last sentence of document as features
    #corp.remove_whitespace()
    #corp.lower()          #lowers all characters
    #corp.stopword_feature()
    #print(corp.df)
    quit()

    # Stemming. Seems to work ok. not necessary soufyan made a nice lemmatrizer
    stemmed = Corpus(data).stem().df
    print(stemmed)

    # Numerical char replacement. Doesn't work.
    num_replace = Corpus(data).replace_num().df
    print(num_replace.head(100))

    # lowercasing. Seems to work.
    lowercased = Corpus(data).lower().df
    print(lowercased)


    # print(pd.concat([sent_tagged, lowercased, stemmed, num_replace], axis=1))
