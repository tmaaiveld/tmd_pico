# import keras
from pathlib import Path
from util import c, get_id
import pandas as pd
import preprocessing as pre


RAW_DATA_PATH = Path('data/raw')
STANFORD_DATA_PATH = Path('data/CoNNL')

# todo: not implemented, see raw() in preprocessing
raw_df = pre.import_data(RAW_DATA_PATH, 'raw.csv', preprocessing=pre.raw)

# to run, drop ebm_1_parsed into /data/CoNNL (ebm_2_parsed is optional for now)
# stanford_df = pre.import_data(STANFORD_DATA_PATH, 'connl.csv', preprocessing=pre.CoNNL)

# print(f'data directory: {stanford_df}')

# df = pre.merge() # todo: not implemented

# df = pd.read_csv(stanford_df, header=0, index_col=['PMID', 'token']).drop('ID', axis=1)

# print(df)
