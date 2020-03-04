"""
- see design.txt for what needs to be implemented here. w2v_pipeline.ipynb has some minimal examples.
- take a look at raw_vocab.csv to see the tokens sorted by commonality (run preprocessing.py first).

"""

import decimal as dec
from pathlib import Path
import pandas as pd
import nltk

# nltk.download('averaged_perceptron_tagger')

RAW_PKL = Path('data/raw.pkl')
TEMP = Path('data/temp')


def lemmatize(df):
    """
    Converts raw imported word data from EBM-NLP to processed lemma tags.
    Lowercase -> handle numeric chars -> handle punctuation chars

    todo: needs improvement/finishing
    """

    print(f'{len(df.Words.value_counts())} unique values detected before processing')

    # todo: may need to deliberate more on this step.
    [token.lower() for token in df.Words] # case handling

    # should check out raw_vocab.csv some more. See raw_vocab_vis.ipynb
    # try to detect units using regex matching etc?

    df.lemma = df.lemma.apply(handle_numeric)
    df.lemma = df.lemma.apply(handle_punctuation)

    # additional steps?

    print(f'{len(df.Words.value_counts())} unique values remaining after processing')

    return df

def handle_numeric(lemma):
    """
    Check if a token is numeric. Numbers with units and a-mods ('2-fold') are not changed.
    Full numeric ('25') is handled differently from decimal ('-0.1'). This is to distinguish between counts and
    measurements/variables/hyperparameters in the text.
    """

    if lemma.isnumeric():
        return '<NUM>'

    else:
        try:
            dec.Decimal(lemma)
            return '<DEC>'
        except dec.InvalidOperation:
            return lemma


def handle_punctuation(lemma):
    """
    Handle punctuation marks for the ground truth data. Periods and commas should be fine to leave, but perhaps certain
    chars should be tagged with a more generic label. Take a look at the top tokens in raw_vocab.csv and read
    'selective removal of certain embeddings/punctuations' in
    https://www.quora.com/Why-should-punctuation-be-removed-in-Word2vec.
    """

    raise NotImplementedError()


def w2v_prep(df):
    """
    Prepares the data for ge

    - removing sentence punctuation (different from labeled data - replace . with '\n' to learn word embeddings)
    - note that the model requires some information on sentence demarcation in the training data.

    read these steps before making changes: https://www.quora.com/Why-should-punctuation-be-removed-in-Word2vec
    """

    raise NotImplementedError()


def main():

    df = pd.read_pickle(RAW_PKL)
    df.lemma = lemmatize(df.Words)

    # handling punctuation
        # for word2vec,

    # POS tagging
    df['POS'] = nltk.pos_tag(df['Word'].to_list())

    # w2v
    # generate processed_vocab

    vocab = df['lemma'].value_counts()

    # drop the w2v input to a text file for model input
    w2v_data = TEMP / 'w2v_data.txt'
    df.w2v_input = w2v_prep(df)

    with w2v_data.open() as f:
        lines = df['w2v_input'].to_list()
        for line in lines:
            f.write(line)
        f.close()

    # save the w2v input

    # import w2v function and generate lookup table
    # construct features from the table (probably cluster label)

    # vocab

    # POS tagging

    # lag columns? for which features?

    # add more features? Some can be easily added using nltk



if __name__ == '__main__':

    test_chars = ['1','-1','0','X','0.01','-0.01', '50mg Hg', '2mg/L', 'hi', 'aesthetics']
    for c in test_chars:
        print(f'{c} -> {handle_numeric(c)}')

    # main()

