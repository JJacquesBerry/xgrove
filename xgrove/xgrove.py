import sklearn
import sklearn.datasets
import sklearn.metrics as metrics
import sklearn.model_selection
import sklearn.tree as tree
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import graphviz
import os
import statistics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from pandas import read_csv
from sklearn.ensemble import HistGradientBoostingRegressor
    

# read testing dataset
data = read_csv(r'C:\Users\jjacq\xgrove\data\HousingData.csv')

# create dataframe 
df = pd.DataFrame(data)

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

class xgrove():
    # define xgrove-class with default values
    # TODO add type check
    def __init__(self, 
                 model, 
                 data: pd.DataFrame, 
                 ntrees: np.array = np.array([4, 8, 16, 32, 64, 128]), 
                 pfun = None, 
                 shrink: int = 1, 
                 b_frac: int = 1, 
                 seed: int = 42,
                 grove_rate: float = 0.1):
        self.model = model,
        self.data = data,
        self.ntrees = ntrees,
        self.pfun = pfun,
        self.shrink = shrink,
        self.b_frac = b_frac,
        self.seed = seed,
        self.grove_rate = grove_rate
        self.surrTar = self.getSurrogateTarget()
        self.surrGrove = self.getSurrogateGrove()

# get-functions for class overarcing variables
    def getSurrogateTarget(self):
        target = self.model.predict(self.data)
        return target
    
    def getSurrogateGrove(self):
        grove = HistGradientBoostingRegressor(learning_rate=self.grove_rate, random_state=self.seed)
        return grove

    # calculate upsilon
    def upsilon(self, pexp):
        ASE  = statistics.mean((self.surrTar-pexp)**2)
        ASE0 = statistics.mean((self.surrTar-statistics.mean(self.surrTar))**2)
        ups = 1- ASE/ASE0
        return ups
# define plot method for the Upsilon-Rules-Curve of the surrogate grove
    def plot(x,
             abs = "rules",
             ord = "upsilon"):
        abs = abs
        ord = ord
        plt.plot()