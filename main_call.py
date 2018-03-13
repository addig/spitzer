"""
This program builds a model to classify toxic comments based on the training data.
different models can be used and tested using plug and play technique
"""


# Data Read
# Data Clean
# Feature generation
# Model Building
# Model validation
# Test
import read_data as r_d
import clean_data as c_d
import feature_data as f_d
import model_build as m_b
import model_test as m_t
import numpy as np
import pandas as pd

# File names
train_name = 'train.csv'
test_name = 'test.csv'
model_name = 'svm'
#model_name = 'logistic'
target = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']


#Call data read
train_data = r_d.Data_Read(train_name)
test_data = r_d.Data_Read(test_name)
pred_matrix = np.zeros((test_data.data_df.shape[0],len(target)))
print 1
clean_train_data = c_d.Data_Clean(train_data.data_df)
clean_test_data = c_d.Data_Clean(test_data.data_df)

features_train_data = f_d.Feature_generator(clean_train_data.data_df)
features_test_data = f_d.Feature_generator(clean_test_data.data_df)

for i, col in enumerate(target):
    model_build = m_b.Model_builder(features_train_data.data_df, features_train_data.tfidf_matrix,model_name,i, col)
    test_model = m_t.Model_Tester(features_test_data.data_df, features_test_data.tfidf_matrix,model_build.model, model_build.model_name,i, col)
    pred_matrix [:,i] = test_model.pred_matrix
print 1
prediction = pd.DataFrame(pred_matrix, columns=target)
prediction.insert(loc=0, column='id', value=features_test_data.data_df['id'])
prediction.to_csv("result.csv", index= False)
print 1