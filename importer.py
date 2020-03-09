import tarfile
import zipfile
import shutil
from pathlib import Path
import pandas as pd
from util import c, get_id, make_dirs

# these will end up in a params dict somewhere
DATA_PATH = Path('data')
# STANFORD_COLS = ["Sent_pos", "Word", "Lemma", "UPOS", "XPOS", "HEAD", "DEPREL"]
PHASE = 'starting_spans' # 'hierarchical_labels'
CATS = ['interventions', 'outcomes', 'participants']


def import_raw(path, phase, verbose=True):
    """
    Function to extract EBM-NLP data directly from tarfile.
    """
    temp = Path('data/temp')
    name = path.name.split('.')[0]

    try:
        tar = tarfile.open(path.resolve())
    except FileNotFoundError:
        raise FileNotFoundError('Place the tarfile in the /data/raw directory.')

    if verbose: print(f'Extracting {path.name}. This may take a few minutes.')

    tar.extractall(path=temp)
    temp = temp / name # use tmpfile?

    doc_fnames = list((temp / 'documents/').glob('*.tokens'))

    if verbose: print(f"parsing {len(doc_fnames)} documents from {path / 'documents/'}.")

    tokens = {f.name.split('.')[0]: f.open(encoding='latin-1').read().split() for f in doc_fnames}
    token_s = prep_tokens(tokens)
    token_s.name = 'Word'
    token_s.index.names = ['doc', 'idx']
    token_s.value_counts().to_csv('data/raw_vocab.csv')

    pio = dict.fromkeys(CATS)
    n_imported_rows = 0

    for cat in pio:
        train_fnames = list(temp.glob(f'annotations/aggregated/{phase}/{cat}/train/*.ann'))
        test_fnames = list(temp.glob(f'annotations/aggregated/{phase}/{cat}/test/crowd/*.ann'))

        # split parameters differ for ebm_nlp_1_00 ['_'] and ebm_nl_2_00
        # split_params = ['_', ','] if path.name == 'ebm_nlp_1_00' else ['.','\n']

        pio[cat] = {f.name.split('_')[0]: f.open().read().split(',')
                    for f in train_fnames + test_fnames}

        n_imported_rows += sum([len(l) for l in pio[cat].values()])

        print(f'Successfully parsed {cat}.')

    tar.close()
    shutil.rmtree(temp)

    # token_s = token_s.loc[(list(pio[cat].keys()),slice(None))] # todo: necessary for ebm_nlp_2

    try:
        assert len(token_s) == n_imported_rows / len(pio)

    except AssertionError:
        print(f'length of token series: {len(token_s)}')
        print(f'number of imported label rows: {n_imported_rows / len(pio)}')
        raise AssertionError('Merge failed. Difference detected in the number of tokens and labels.\n',
                             'Note: this should not be considered an error for ebm_nlp_2_00.tar.gz')

    labels_df = pd.concat(
        [pd.DataFrame.from_dict(pio[cat], orient='index').stack() for cat in pio],
        axis=1)

    labels_df.columns = CATS
    labels_df.index.names = ['doc', 'idx']

    df = pd.merge(token_s, labels_df, how='inner', on=['doc','idx']) # 'inner' for ebm_nlp_2, differing number of docs

    for col in df.columns:
        df[col] = df[col].astype(int, errors='ignore')

    if verbose:
        print(f'Imported {len(df.index)} tokens.')
        print(f'{df.isna().sum().sum()} values were missing from the labels.')

    return df


def prep_tokens(tok_dict):
    s = pd.DataFrame.from_dict(tok_dict, orient='index').stack()

    return s


if __name__ == '__main__': # todo: move contents once imported

    make_dirs(*[(DATA_PATH / d) for d in ['raw', 'CoNNL', 'temp', 'ft_models']])

    if (DATA_PATH / 'labels.pkl').exists():
        raw = pd.read_pickle(DATA_PATH / 'labels.pkl')
        print(f"Importing data from {DATA_PATH / 'labels.pkl'}")

    else:
        try:
            raw = import_raw(DATA_PATH / 'raw/ebm_nlp_1_00.tar.gz', PHASE, verbose=True)
            # stanford = import_CoNNL(DATA_PATH / 'CoNNL', verbose=False)

            assert len(raw) > 0 # and len(stanford) > 0

        except AssertionError:
            raise Exception('Import failed. Ensure ebm_nlp_*_00.tar.gz is placed in data/raw.')

    doc_ids = len(raw.index.unique(level='doc'))

    raw.to_pickle('data/labels.pkl')
    print('Preprocessing complete.')
