import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')

def tfidf(sentences):

    tfidf_matrix = vectorizer.fit_transform(sentences)
    word_index = vectorizer.vocabulary_
    feature_names = vectorizer.get_feature_names()
    X = tfidf_matrix.todense()

    mean_tfidf = pd.Series(np.nanmean(X, axis=0).tolist()[0],
                           index=feature_names, name='TF-IDF Table')

    return mean_tfidf


if __name__ == '__main__':
    docs = pd.read_pickle('data/sentences.pickle')
    result = tfidf(docs)

    #document index
    # doc = 4
    # print(list(joined_docs.values())[doc])

    # get all indexes for the words in the doc
    # doc_feature_index = tfidf_matrix[doc,:].nonzero()[1]

    # get the tfidf scores for those word indexes
    # tfidf_scores = zip(doc_feature_index, [tfidf_matrix[doc, x] for x in doc_feature_index])

    # for word, score in [(feature_names[index], score) for (index, score) in tfidf_scores]:
    #     print(word, score)

    # df = pd.DataFrame(tfidf_matrix.toarray(), columns = vectorizer.get_feature_names(), index=docs.keys())
    #
    # # to see the value of a word in all the docs (just an example)
    # print(df[df['prevention']!=0]['prevention'])
    #
    # means = np.nanmean(X, axis=0)
    # means = dict(zip(feature_names, means.tolist()[0]))
    #
    # # mean tfidf score for a word.
    # print(means['outcome'])