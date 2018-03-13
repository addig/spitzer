import string

class Data_Clean(object):

    """
    Clean the imput data
    """
    def __init__(self,data_df):
        self.data_df = data_df
        self.clean_file()

    def clean_file(self):
        self.data_df['clean_text'] = self.data_df['comment_text'].str.lower()
        self.data_df['clean_text'] = self.data_df['clean_text'].str.strip()
        self.data_df.loc[:, 'clean_text'] = (self.data_df['clean_text'].apply(lambda p:p.translate(None, string.punctuation)))



