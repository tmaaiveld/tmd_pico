import sys
import tarfile
import shutil
from pathlib import Path
import pandas as pd

DATA_PATH = Path('../data/')
PHASE = 'starting_spans'
CATS = ('interventions', 'outcomes', 'participants')


def load_tarfile(path, phase=PHASE, extract_labels=CATS, verbose=True):

    temp = DATA_PATH / 'temp'
    name = path.name.split('.')[0]

    try:
        tar = tarfile.open(path.resolve())
    except FileNotFoundError:
        raise FileNotFoundError('Place the tarfile in the /data/raw directory.')

    if verbose: print(f'Extracting {path.name}. This may take a few minutes.')

    tar.extractall(path=temp)
    temp = temp / name

    doc_fnames = list((temp / 'documents/').glob('*.tokens'))

    if verbose: print(f"parsing {len(doc_fnames)} documents from {path / 'documents/'}.")

    tokens = {f.name.split('.')[0]: f.open(encoding='latin-1').read().split() for f in doc_fnames}
    token_s = prep_tokens(tokens)
    token_s.name = 'Word'
    token_s.index.names = ['doc', 'idx']
    token_s.value_counts().to_csv(str(DATA_PATH / 'raw_vocab.csv'))

    pio = dict.fromkeys(extract_labels)
    n_imported_rows = 0

    for cat in pio:
        train_fnames = list(temp.glob(f'annotations/aggregated/{phase}/{cat}/train/*.ann'))
        test_fnames = list(temp.glob(f'annotations/aggregated/{phase}/{cat}/test/gold/*.ann'))

        # split parameters differ for ebm_nlp_1_00 ['_'] and ebm_nl_2_00
        # split_params = ['_', ','] if path.name == 'ebm_nlp_1_00' else ['.','\n']

        pio[cat] = {f.name.split('_')[0]: f.open().read().split(',')
                    for f in train_fnames + test_fnames}

        n_imported_rows += sum([len(l) for l in pio[cat].values()])

        print(f'Successfully parsed {cat}.')

    tar.close()
    shutil.rmtree(temp) # todo: doesn't work

    try:
        assert len(token_s) == n_imported_rows / len(pio)

    except AssertionError:
        print(f'length of token series: {len(token_s)}')
        print(f'number of imported label rows: {n_imported_rows / len(pio)}')
        raise AssertionError('Merge failed. Difference detected in the number of tokens and labels.\n',
                             'Importing ebm_nlp_2 or other non-matching files not implemented.')
    # token_s = token_s.loc[(list(pio[cat].keys()),slice(None))] # todo: necessary for ebm_nlp_2. Could do with merge -> dropna?

    labels_df = pd.concat(
        [pd.DataFrame.from_dict(pio[cat], orient='index').stack() for cat in pio],
        axis=1)

    labels_df.columns = CATS
    labels_df.index.names = ['doc', 'idx']

    df = pd.merge(token_s, labels_df, how='inner', on=['doc','idx']) # 'inner' for ebm_nlp_2, differing number of docs

    for col in df.columns:
        df[col] = df[col].astype(int, errors='ignore')

    if verbose:
        print(f'Imported {len(df.index)} tokens from {path.name}.')
        print(f'{df.isna().sum().sum()} values were missing from the labels.')

    return df


def prep_tokens(token_dict):
    return pd.DataFrame.from_dict(token_dict, orient='index').stack()


if __name__ == '__main__':

    DATA_PATH.mkdir(exist_ok=True, parents=True)

    if (DATA_PATH / 'labels.parquet').exists():
        data = pd.read_parquet(DATA_PATH / 'labels.parquet')
        print(f"Importing data from {DATA_PATH / 'labels.parquet'}")

    else:
        try:
            data = load_tarfile(Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_PATH / 'ebm_nlp_1_00.tar.gz',
                                PHASE, verbose=True)
            assert len(data) > 0

        except AssertionError:
            raise Exception(f'Import failed. Ensure ebm_nlp_*_00.tar.gz is placed in {DATA_PATH}')

        data.to_parquet(str(DATA_PATH / 'labels.parquet'))
        print(f"Preprocessing complete. Data saved to {str(DATA_PATH / 'labels.parquet')}.")
