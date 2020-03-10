import pandas as pd
import decimal as dec
from nltk.stem.snowball import SnowballStemmer

class Corpus: # move later
    """A class which receives a pd.Series of indexed documents and can execute various preprocessing methods to produce different version of word columns or different features to be added."""

    # add methods to create processing protocols for data
    # lowercasing

    # special chars are best left unhandled as they are strong predictors
    # could replace uncommon chars completely, at least for the W2V columns.
    # any preprocessing that doesn't remove instances is fine, and can be used to generate more features

    # https://radimrehurek.com/gensim/parsing/preprocessing.html
    # https://www.nltk.org/_modules/nltk/parse/stanford.html <- can try stanford again using sep sent tokens?

    def __init__(self, doc_series):
        self.doc_series = doc_series # remains unchanged
        self.df = doc_series.to_frame() # is changed
        self.idx = doc_series.index
        self.doc_lists = doc_series.groupby('doc').apply(list).tolist()
        self.sent_lists = None
        
        print(self.df)
        quit()

    # modify series.name as
    
    def return_words_nparr(self):
        return self.df.values.tolist()
        
    def process(self):
        self.idx_sentences()
        self.lower()


    def idx_sentences(self): # todo: could be cleaned up
        df = self.df

        sent_break = df['Word'].mask(df['Word'] == '.', 1).mask(df['Word'] != '.', 0)  \
                               .droplevel(level='idx').fillna(method='bfill')

        sent_id = sent_break.groupby('doc').shift(1).fillna(method='bfill')  \
                            .groupby('doc').expanding().sum()                \
                            .astype(int).rename('sent_id')

        self.df['sent_id'] = sent_id.values
        self.df = self.df.set_index('sent_id', append=True).reset_index(level='idx')

        # self.df['sent_id'] = sent_id.values # set to int for now

        print(sent_id.values)
        print(self.df)
        quit()


        sent_pos = pd.Series(1, index=df.index).groupby(['doc','sent_id']).expanding().sum()

        # print(df)
        # sent_pos = df['dummy']#.groupby(['doc','sent_id']).expanding().sum()
        print(sent_pos)
        quit()

        self.sentences = df['Word'].groupby(['doc','sent_id']).apply(list)

        self.df['sent_pos'] = sent_pos.reset_index(drop=True)

        # print(sent_pos)
        # self.sentences = sent_pos
                                         # .droplevel(level='sent_id').set_index('idx', append=True)



        # print(self.sentences)


        # todo: test

        # print(df)

        return self


    def by_sentence(self):
        try:
            self.df = self.df.set_index('sent_id', append=True).reset_index(level='idx')

        except KeyError:
            print('sent_id index not found. Attempting to index sentences.')
            self.idx_sentences()

        return self



    def sent_pos(self):
        self.df['sent_pos'] = self.sentences.groupby('doc', 'sent_id').expanding().sum()


    def lower(self): # todo: test

        self.df['Word'] = [word.lower() for word in self.df['Word']]

        return self

    def stem(self):

        self.df['Word'] = [SnowballStemmer("english", ignore_stopwords=True).stem(word)
                           for word in self.df['Word']]
        return self

    def replace_num(self):
        """
        Check if a token is numeric. Numbers with units and a-mods ('2-fold') are not changed.
        Full numeric ('25') is handled differently from decimal ('-0.1'). This is to distinguish between counts and
        measurements/variables/hyperparameters in the text.
        """
        self.df['Words'] = self.df['Words'].apply(self.handle_numeric)

        return self

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

    # preprocessing by method chaining
    # see deprecated for some suggestions

    # write methods for...
        # lowercase
        # special char removal (improves POS)
        # pos tagging? or keep in pipeline?
        # stemming
        # numchar replace (see depr)

    # try deprel with Stanford once again (use nltk.parse.stanford.parse_sents

    # pos_replace = {
    #     '$': 'SYM',
    #     '#': 'SYM',
    # }
    # features['POS'] = features['POS_NLTK'].replace(['$', '#'], 'SYM')

    # could test other postaggers if they can tag a list
    # add nltk tnt pos tag < tests
    # add gensim pos tag < tests


    def zip_docs(self):
        self.df.groupby('doc').apply(list).to_dict() # list of lists output for zip methods

    def zip_sents(self):
        self.df.groupby('doc', 'sentence').apply(list).to_dict()

    # def __iter__(self): # todo: fix or remove
    #     for line in open(self.pkl): # see function below for file building
    #         yield utils.simple_preprocess(line) # doesn't work yet because of stop word removal etc.


if __name__ == '__main__':

    # dataset production demo

    # could use parent/child (DocSeries parent, Corpus child)
    print('loading data')
    data = pd.read_parquet('data/split/train_1000.parquet')['Word']
    corp = Corpus(data[1000:])

    print(corp)

    num_replace = Corpus(data).replace_num() # Important, but may already work
    print(num_replace)
    
    sent_tagged = corp.idx_sentences().df
    print(sent_tagged)

    lowercased = Corpus(data).lower().df
    print(lowercased)

    stemmed = Corpus(data).stem()
    print(stemmed)



    print(pd.concat([sent_tagged, lowercased, stemmed, num_replace], axis=1))



