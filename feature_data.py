from sklearn.feature_extraction.text import TfidfVectorizer


class Feature_generator(object):

    """
    Generate the features
    """
    def __init__(self,data_df):
        self.data_df = data_df
        self.TFIDF_generator()

    def TFIDF_generator(self):
        tfidf = TfidfVectorizer(
            min_df=1,  # min count for relevant vocabulary
            max_features=10000,  # maximum number of features
            strip_accents='unicode',  # replace all accented unicode char
            # by their corresponding  ASCII char
            analyzer='word',  # features made of words
            ngram_range=(1, 1),  # features made of a single tokens
            use_idf=True,  # enable inverse-document-frequency reweighting
            smooth_idf=True,  # prevents zero division for unseen words
            sublinear_tf=False,
            stop_words="english",
            lowercase=True)

        # tf-idf matrix of 'comments'
        self.tfidf_matrix = tfidf.fit_transform(self.data_df['clean_text'])
