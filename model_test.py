import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

class Model_Tester(object):

    """
    Generate the features
    """
    def __init__(self,data_df, tfidf_matrix,model, model_name,i,col):
        self.model_name = model_name
        self.model = model
        self.data_df = data_df
        self.col = col
        self.i = i
        self.tfidf_matrix = tfidf_matrix
        self.pred_matrix = np.zeros((self.data_df.shape[0]))
        self.test_model()


    def test_model(self):
        if self.model_name == 'logistic':
            self.pred_matrix = self.model.predict_proba(self.tfidf_matrix)[:, 1]
            print 1
        elif self.model_name == 'svm':
            self.pred_matrix = self.model.predict_proba(self.tfidf_matrix)[:, 1]
            print 2

