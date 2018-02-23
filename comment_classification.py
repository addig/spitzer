import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

# Loading Train and Test Data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Tokenizer initialized
tfidf  = TfidfVectorizer(
            min_df=1,  # min count for relevant vocabulary
            max_features=10000,  # maximum number of features
            strip_accents='unicode',  # replace all accented unicode char
            # by their corresponding  ASCII char
            analyzer='word',  # features made of words
            ngram_range=(1, 1),  # features made of a single tokens
            use_idf=True,  # enable inverse-document-frequency reweighting
            smooth_idf=True,  # prevents zero division for unseen words
            sublinear_tf=False,
            stop_words= "english",
            lowercase=True)

#tf-idf matrix of 'comments'
tfidf_train_matrix = tfidf.fit_transform(train_data['comment_text']);
tfidf_test_matrix = tfidf.fit_transform(test_data['comment_text']);


target = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']
pred_matrix = np.zeros((test_data.shape[0],len(target)));

# Build a Logistic regression model on each Target response variable using
# tf-idf matrix as features to train the model
for i,col in enumerate(target):
    lr = LogisticRegression(random_state=i, class_weight='balanced')
    lr.fit(tfidf_train_matrix ,train_data[col])
    pred_matrix[:,i] = lr.predict_proba(tfidf_test_matrix)[:,1]

# Matrix to Data Frame
prediction = pd.DataFrame(pred_matrix, columns = target)
prediction.insert(loc=0, column='id', value=test_data['id'])
print prediction.head()

#Write to csv
prediction.to_csv("result.csv", index= False)

