"""
- see design.txt for what needs to be implemented here. w2v_pipeline.ipynb has some minimal examples.
- take a look at raw_vocab.csv to see the tokens sorted by commonality (run importer.py first).

"""

from pathlib import Path
import fasttext

DATA = Path('data')
TEMP = DATA / 'temp'
MODEL_PATH = DATA / 'ft_models' / 'BioWordVec_PubMed_MIMICIII_d200.bin'


def load_ft_model(path):
    model = None
    try:
        print('Loading PubMed FastText model.')
        model = fasttext.load_model(str(path))  # for imports with base fastText package

    except MemoryError:
        print('System has run out of memory.')

    print(f'Successfully loaded model from {path}.')
    return model
