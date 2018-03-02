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

# File names
train_name = 'train.csv'
test_name = 'test.csv'


#Call data read
train_data = r_d.Data_Read(train_name)

clean_train_data = c_d.Data_Clean(train_data.data_df)

features_train_data = f_d.Feature_generator(clean_train_data.data_df)
print 1