import re
import scipy.sparse as sp

class textToVect(BaseEstimator):
    def __init__(self, pattern='\w+', stop_words=None):
        self.pattern = pattern
        self.stop_words = stop_words

    def analyzer(self):
        pattern = re.compile(self.pattern)
        stop_words = self.stop_words

        if stop_words is not None:
            return lambda doc: [w for w in pattern.findall(doc.lower()) if w not in stop_words]
        else:
            return lambda doc: pattern.findall(doc.lower())

    def vocabulary_edit(self, texts):
        analyze = self.analyzer()

        texts_words = np.array(list(map(lambda doc: analyze(doc), texts)))
        words = set(sorted(np.hstack(texts_words)))
        n_features = len(words)

        vocabulary = dict(zip(words, range(n_features)))

        return vocabulary

    def count_words(self, texts):
        analyze = self.analyzer()
        vocabulary = self.vocabulary

        j_indices = []
        indptr = []
        values = []
        indptr.append(0)
        for doc in texts:
            feature_counter = {}
            for feature in analyze(doc):
                try:
                    feature_idx = vocabulary[feature]
                    if feature_idx not in feature_counter:
                        feature_counter[feature_idx] = 1
                    else:
                        feature_counter[feature_idx] += 1
                except KeyError:
                    continue

            j_indices.extend(feature_counter.keys())
            values.extend(feature_counter.values())
            indptr.append(len(j_indices))

        j_indices = np.asarray(j_indices)
        indptr = np.asarray(indptr)
        values = np.asarray(values)

        X = sp.csr_matrix((values, j_indices, indptr),
                          shape=(len(indptr) - 1, len(vocabulary)))

        X.sort_indices()
        return np.asarray(X.todense()) #je n'arrive pas à gérer les matrices sparses par la suite

    def fit(self, texts, y=None):
        self.vocabulary = self.vocabulary_edit(texts)

        return self

    def transform(self, texts, y=None):
        X = self.count_words(texts)

        return X
