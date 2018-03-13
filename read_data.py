import pandas as pd


class Data_Read(object):
    """
    Read data from input file
    """
    def __init__(self,filename):
        self.filename = filename
        self.data_df = pd.DataFrame()
        self.readfile()

    def readfile(self):
        self.data_df = pd.read_csv(self.filename)




