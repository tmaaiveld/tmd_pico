This is the repository and code submission for the Automated PICO Extraction project.

The original data set can be downloaded from https://github.com/bepnye/EBM-NLP (ebm_nlp_1_00.tar.gz) and imported by running:

`$ python3 processing/importer.py <path/to/ebm_nlp_1_00.tar.gz>`

- feature_builder.ipynb and train_test_split.ipynb give an overview of the feature extraction and partitioning step. The supplementary code is in the `processing` folder.
- Classification results of the two presented models are in the `results` folder.
- The code to run the final models is in the `classifiers` folder.
