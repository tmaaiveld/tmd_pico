import pandas as pd
import stanfordnlp
import stanfordnlp
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

sentence_deps = []
docs = pd.read_pickle('data/sentences.pickle')

joined_docs = {k:' '.join(v) for k, v in docs.items()}
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(joined_docs.values())
word_index = vectorizer.vocabulary_
feature_names = vectorizer.get_feature_names()

#document index
doc = 0

# get all indexes for the words in the doc
doc_feature_index = tfidf_matrix[doc,:].nonzero()[1]

# get the tfidf scores for those word indexes
tfidf_scores = zip(doc_feature_index, [tfidf_matrix[doc, x] for x in doc_feature_index])


for word, score in [(feature_names[index], score) for (index, score) in tfidf_scores]:
  print(word, score)


df = pd.DataFrame(tfidf_matrix.toarray(), columns = vectorizer.get_feature_names(), index=docs.keys())

dft = df.T
maxValue = dft.max(axis=1)
for i, m in enumerate(maxValue[-10:]):
  tfidf_word    = dft.index[i-10]
  tfidf_score   = m
  
  print(tfidf_word)
  print(tfidf_score)
