import os
import tarfile
import zipfile
from pathlib import Path
from glob import glob
import pandas as pd
from util import c, get_id
import os
import codecs

EBM_COLS = ["ID", "Word", "Lemma", "UPOS", "XPOS", "HEAD", "DEPREL"]


def import_data(path, filename, preprocessing):

    path = Path(path)

    if not path.exists():
        path.mkdir()

        input("Extract EBM archive into the /parsed directory and press enter.")
        # todo: add license and code reference for SCNLP?

    elif (path / filename).exists(): return path / filename

    df = preprocessing(path)

    # todo: module to add labels
    try:
        df.to_csv(path / filename)
    except AttributeError:
        pass

    print('Imported Stanford CoreNLP CoNNL features.')
    return path / filename


def sep_dir(path_to_dir, dir_name):
    dfs = []
    doc_ids = []
    for file_name in path_to_dir.rglob('*.ann'):
        # print(file_name.stem)
        doc_ids.append(get_id(file_name.name))
        with open(path_to_dir / file_name, 'r') as f:
            s = pd.DataFrame(f.read().split(','))
            dfs.append(pd.DataFrame(s))

    doc_keys = c([[doc_ids[i]] * df.shape[0] for i, df in enumerate(dfs)])
    df = pd.concat(dfs)
    df.index = doc_keys
    df.columns = [dir_name]
    return df


def raw(path):
    """
    todo: read tarfile with tarfile module, extract files and preprocess. See
    """
    sep_dfs = []
    folder = 'data/raw/ebm_nlp_1_00/annotations/aggregated/starting_spans/'
    p = Path(os.path.dirname(__file__))
    directories = ['interventions', 'outcomes', 'participants']
    sub_dirs = ['test', 'train']
    for d in directories:
        dir_path = p / folder / d / sub_dirs[1]
        df = sep_dir(dir_path, d)
        sep_dfs.append(df)
    return pd.concat(sep_dfs, axis=1)


def CoNNL(path):

    dfs = []
    doc_ids = []

    for filepath in path.rglob('*.zip'):
        with zipfile.ZipFile(filepath) as z:
            paths = z.namelist()
            print(f'parsing {len(paths)} documents from {filepath}')

            for doc_id in paths:
                doc_ids.append(get_id(doc_id))
                with z.open(doc_id) as f:
                    dfs.append(pd.read_csv(f, delimiter='\t', header=None))

    doc_keys = c([[doc_ids[i]] * df.shape[0] for i, df in enumerate(dfs)])
    tok_keys = c([df.iloc[:,0] for df in dfs])


    index = pd.MultiIndex.from_tuples(zip(doc_keys, tok_keys), names=['PMID', 'token'])

    df = pd.concat(dfs)
    df.index = index
    df.columns = EBM_COLS

    # rewrite index column # todo: not implemented

    return df


def merge():
    """
    Function to align and merge raw tokens and labels into dataframe for keras model
    """
    raise NotImplementedError()
