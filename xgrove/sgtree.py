import mkdocs
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
from sklearn.ensemble import GradientBoostingRegressor

class sgtree():
    def __init__(self, 
                 model, 
                 data: pd.DataFrame, 
                 maxdeps: np.array = np.array(range(1,9)), 
                 cparam = 0,
                 pfun = None
                 ):
        self.model = model
        self.data = self.encodeCategorical(data)
        self.maxdeps = maxdeps
        self.cparam = cparam
        self.pfun = pfun
        self.surrTar = self.getSurrogateTarget(pfun)
        self.surrogate_trees = []
        self.rules = []
        self.explanation = []

    def encodeCategorical(self, data):
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        data_encoded = pd.get_dummies(data, columns=categorical_columns)
        return data_encoded

    def getSurrogateTarget(self, pfun):
        if self.pfun is None:
            self.surrTar = self.model.predict(self.data)
        else:
            self.surrTar = pfun(model=self.model, data=self.data)
        if not isinstance(self.surrTar, np.ndarray) or len(self.surrTar.shape) != 1:
            raise ValueError("pfun does not return a numeric vector!")
        return self.surrTar

    def upsilon(self, pexp):
        ASE  = statistics.mean((self.surrTar - pexp) ** 2)
        ASE0 = statistics.mean((self.surrTar - statistics.mean(self.surrTar)) ** 2)
        ups = 1 - ASE / ASE0
        rho = statistics.correlation(self.surrTar, pexp)
        return ups, rho

    def calcusatesgtree(self):
        for md in self.maxdeps:
            model = tree.DecisionTreeRegressor(max_depth=md, ccp_alpha=self.cparam, min_samples_split=2, min_samples_leaf=1).fit(X=self.data, y=self.surrTar)
            t = model.tree_
            features = t.feature
            thresholds = t.threshold
            rules = []
            
            for node in range(t.node_count):
                ncat = []
                if features[node] != -2:  # Check if it's not a leaf node
                    if pd.api.types.is_string_dtype(self.data.iloc[:, features[node]]):
                        ncat.append(len(self.data.iloc[:, features[node]].unique()))
                    else:
                        ncat.append(-1)

                    rule = {
                        'feature': features[node],
                        'threshold': thresholds[node],
                        'pleft': t.value[t.children_left[node]][0][0],
                        'pright': t.value[t.children_right[node]][0][0],
                        'ncat': ncat[0]
                    }
                    rules.append(rule)

            rules_df = pd.DataFrame(rules)
            self.surrogate_trees.append(rules_df)

            # Predict values for the current surrogate tree
            predictions = model.predict(self.data)
            
            # Call the upsilon method to compute upsilon and rho
            upsilon_val, rho_val = self.upsilon(predictions)

            explanation_entry = {
                "trees": 1,
                "rules": len(rules),
                "upsilon": upsilon_val,
                "cor": rho_val
            }
            self.explanation.append(explanation_entry)

    # Method to get the results (explanation, rules, and surrogate_trees)
    def get_results(self):
        results = {
            "explanation": pd.DataFrame(self.explanation),
            "rules": self.rules,
            "surrogate_trees": self.surrogate_trees
        }
        return results
        print(self.explanation)
