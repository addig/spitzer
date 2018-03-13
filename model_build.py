from sklearn.linear_model import LogisticRegression
from sklearn import svm

class Model_builder(object):

    """
    Generate the features
    """
    def __init__(self,data_df,tfidf_matrix,model_name,i,col):
        self.tfidf_train_matrix = tfidf_matrix
        self.model_name = model_name
        self.data_df = data_df
        self.col = col
        self.i = i
        self.train_model()

    def train_model(self):
        if self.model_name == 'logistic':
            self.model = LogisticRegression(random_state=self.i, class_weight='balanced')
            self.model.fit(self.tfidf_train_matrix, self.data_df[self.col])

            print 1
        elif self.model_name == 'svm':
            self.model = svm.SVC()
            self.model.fit(self.tfidf_train_matrix, self.data_df[self.col])
            # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
            #     decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
            #     max_iter=-1, probability=False, random_state=None, shrinking=True,
            #     tol=0.001, verbose=False)
            print 2


