from stanfordcorenlp import StanfordCoreNLP
import logging
import json
import os
from pathlib import Path


class StanfordNLP:
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port,
                                   timeout=30000)  # , quiet=False, logging_level=logging.DEBUG)
        self.props = {
            'annotators': 'depparse',
            'pipelineLanguage': 'en',
            'outputFormat': 'conll',
            'timeout': 30000
        }

    def word_tokenize(self, sentence):
        return self.nlp.word_tokenize(sentence)

    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)

    def ner(self, sentence):
        return self.nlp.ner(sentence)

    def parse(self, sentence):
        return self.nlp.parse(sentence)

    def dependency_parse(self, sentence):
        return self.nlp.dependency_parse(sentence)

    def annotate(self, sentence):
        # return json.loads(self.nlp.annotate(sentence, properties=self.props))
        return self.nlp.annotate(sentence, properties=self.props)

    @staticmethod
    def tokens_to_dict(_tokens):
        tokens = defaultdict(dict)
        for token in _tokens:
            tokens[int(token['index'])] = {
                'word': token['word'],
                'lemma': token['lemma'],
                'pos': token['pos'],
                'ner': token['ner']
            }
        return tokens


if __name__ == '__main__':
    dir_path = Path('data/CoNNL-NLP/ebm_nlp_1_00/documents')
    sNLP = StanfordNLP()
    text_files = []
    for f in os.listdir(dir_path):
        if f.endswith('.text'):
            text_files.append(f)
    for file_name in text_files[:1]:
        print(f'PARSING {file_name}')
        full_path = os.path.join(dir_path, file_name)
        raw_text = open(full_path, 'r').read()
        if not raw_text:
            print('NOTHING HERE')
            continue
        p = sNLP.annotate(raw_text)
        print(p)
        # if not p:
        #     print("NOTHING HAS BEEN RETURNED BY PARSER")
        #     continue
        # with open(full_path+".parsed", 'w') as f:
        #     f.write(p)
