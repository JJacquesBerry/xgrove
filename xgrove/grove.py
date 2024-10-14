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
    

# read testing dataset
# data = read_csv(r'C:\Users\jjacq\xgrove\data\HousingData.csv')

# # create dataframe 
# df = pd.DataFrame(data)

# TODO: delete direct directory reference
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

class grove():
    # define xgrove-class with default values
    # TODO add type check
    print("its upgraded")
    def __init__(self, 
                 model, 
                 data: pd.DataFrame,
                 surrTarName: str, 
                 ntrees: np.array = np.array([4, 8, 16, 32, 64, 128]), 
                 pfun = None, 
                 shrink: int = 1, 
                 b_frac: int = 1, 
                 seed: int = 42,
                 grove_rate: float = 1,
                 ):
        self.model = model
        self.data = self.encodeCategorical(data)
        self.surrTarName = surrTarName
        self.ntrees = ntrees
        self.pfun = pfun
        self.shrink = shrink
        self.b_frac = b_frac
        self.seed = seed
        self.grove_rate = grove_rate
        self.surrTar = self.getSurrogateTarget(pfun = self.pfun)
        self.surrGrove = self.getGBM()
        self.explanation = []
        self.groves = []
        self.rules = []
        self.result = []

    # get-functions for class overarching variables
    def getSurrogateTarget(self, pfun):

        if self.pfun is None:
            self.surrGrove.fit(self.data.drop(self.surrTarName, axis=1), self.data[self.surrTarName])        
            target = self.surrGrove.predict(self.data.drop(self.surrTarName, axis=1))
        else:
            # potentielle Fehlerquelle
            target = pfun(model=self.model, data=self.data)
        return target
    
    def getGBM(self):
        grove = GradientBoostingRegressor(n_estimators=self.ntrees,
                                          learning_rate=self.shrink,
                                          subsample=self.b_frac)
        return grove

    # OHE for evaluating categorical columns
    def encodeCategorical(self, data):
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        data_encoded = pd.get_dummies(data, columns=categorical_columns)
        return data_encoded

    # calculate upsilon
    def upsilon(self, pexp):
        ASE = statistics.mean((self.surrTar - pexp) ** 2)
        ASE0 = statistics.mean((self.surrTar - statistics.mean(self.surrTar)) ** 2)
        ups = 1 - ASE / ASE0
        rho = statistics.correlation(self.surrTar, pexp)
        return ups, rho

    def get_result(self):
        res = [self.explanation, self.rules, self.groves, self.model]
        return res
    
    # Plot method to visualize Upsilon vs. Rules
    def plot(self, abs="rules", ord="upsilon"):
        if len(self.explanation) == 0:
            raise ValueError("No explanation data available. Please run the calculation first.")
        
        # Get the corresponding indices for the given abs (x-axis) and ord (y-axis)
        x_col = self.explanation[abs] if abs in self.explanation.columns else None
        y_col = self.explanation[ord] if ord in self.explanation.columns else None
        
        if x_col is None or y_col is None:
            raise ValueError(f"Cannot find '{abs}' or '{ord}' in explanation columns.")
        
        # Plot the x and y values
        plt.plot(x_col, y_col, marker='o', linestyle='-', color='b')
        plt.xlabel(abs)
        plt.ylabel(ord)
        plt.title(f'{ord} vs {abs}')
        plt.grid(True)
        plt.show()
    def calculateGrove(self):
        explanation = []
        groves = []
        interpretation = []

        # for every tree
        for nt in self.ntrees:
            # predictions generation
            predictions = self.surrGrove.staged_predict(self.data)
            predictions = [next(predictions) for _ in range(nt)][-1]

            rules = []
            for tid in range(nt):
                # extract tree
                tree = self.surrGrove.estimators_[tid, 0].tree_
                # iterate every node of the tree
                for node_id in range(tree.node_count):
                    if tree.children_left[node_id] != tree.children_right[node_id]:  #  splitsnode
                        # save rule
                        rule = {
                            'feature': tree.feature[node_id],
                            'threshold': tree.threshold[node_id],
                            'pleft': tree.value[tree.children_left[node_id]][0][0],
                            'pright': tree.value[tree.children_right[node_id]][0][0]
                        }
                        rules.append(rule)
            
            # convert to dataframe and add to rules
                rules_df = pd.DataFrame(rules)
                groves.append(rules_df)
            
            vars = []
            splits= []
            csplits_left = []
            pleft = []
            pright = []
            for i in range(len(rules_df)):
                vars = vars.append(data.columns[rules])
                feature_index = rules_df.iloc[i]['feature']
                var_name = rules_df.columns[feature_index]
                # Categorical columns
                
######################### Potentielle Fehlerquelle ####################################

                if rules_df.columns[i].dtype == pd.Categorical:
                    levs = rules_df.columns[i].cat.categories
                    lids = self.surrGrove.estimators_[0, 0].tree_.value[int(rules_df.iloc[i]['threshold'])] == -1
                    if sum(lids) == 1: levs = levs[lids]
                    if sum(lids) > 1: levs = " | ".join(levs[lids])
                    csl = levs[0] if isinstance(levs, (list, pd.Index)) else levs
                    if len(levs) > 1:
                        csl = " | ".join(levs)

                    splits.append("")
                    csplits_left.append(csl)

                elif pd.api.types.is_string_dtype(rules_df.columns[i]) or rules_df.columns[i].dtype == object:
                    #print(i+": Kategorisch")
                    levs = rules_df.columns[var_name].unique()
                    lids = self.surrGrove.estimators_[0, 0].tree_.value[int(rules_df.iloc[i]['threshold'])] == -1
                    if sum(lids) == 1: levs = levs[lids]
                    if sum(lids) > 1: levs = " | ".join(levs[lids])
                    csl = levs[0] if isinstance(levs, (list, pd.Index)) else levs
                    if len(levs) > 1:
                        csl = " | ".join(levs)

                    splits.append("")
                    csplits_left.append(csl)
                
                # Numeric columns   
                elif pd.api.types.is_numeric_dtype(rules_df.columns[i]) or np.issubdtype(rules_df.columns[i].dtype, np.number):
                    #print(i+": Numerisch")
                    splits = splits.append(rules_df.iloc[i]["threshold"])
                    csplits_left.append(pd.NA)

                else:
                    print(rules_df.columns[i]+": uncaught case")
            # rules filled
            pleft.append(rules_df[i]["pleft"])
            pright.append(rules_df[i]["pleft"])
        
            basepred = self.surrGrove.estimator_
            df = pd.DataFrame({
                "vars": vars,
                "splits": splits,
                "left": csplits_left,
                "pleft": round(pleft, 4),
                "pright": round(pright, 4)
            })
            df = df.groupby(vars, splits, left)
            df_small = df.agg({"pleft" : "sum", "pright" : "sum"})

            if(len(df_small) > 1):
                i = 2
                while (i != 0):
                    drop_rule = False
                    # check if its numeric AND NOT categorical
                    if pd.api.types.is_numeric_dtype(rules_df.columns[i]) or np.issubdtype(rules_df.columns[i].dtype, np.number) and not(rules_df.columns[i].dtype == pd.Categorical or pd.api.types.is_string_dtype(rules_df.columns[i]) or rules_df.columns[i].dtype == object):
                        #print(i+": Numerisch")
                        for j in range(0, i):
                            if df_small.vars[i] == df_small.vars[j]:
                                v1 = data[df_small.vars[i]] <= df_small.splits[i]
                                v2 = data[df_small.vars[j]] <= df_small.splits[j]
                                tab = [v1,v2]
                                if sum(np.diag(tab)) == sum(tab):
                                    df_small.pleft[j]  = df_small.pleft[i] + df_small.pleft[j] 
                                    df_small.pright[j] = df_small.pright[i] + df_small.pright[j] 
                                    drop_rule = True
                    if drop_rule: df_small = df_small[-i]
                    if not drop_rule: i = i+1
                    if i > len(df_small): i = 0
            # compute complexity and explainability statistics
            upsilon, rho = self.upsilon()

            df0 = pd.DataFrame({
                "vars": "Interept",
                "splits": pd.NA,
                "left": pd.NA,
                "pleft": basepred,
                "pright": basepred
            })
            df = pd.concat([df0, df], ignore_index=True)
            df_small = pd.concat([df0, df_small], ignore_index = True)

            # for better
            df = df.rename({
                "vars": "variable",
                "splits": "upper_bound_left",
                "left": "levels_left"
                }, axis=1) 
            df_small = df_small.rename({
                "vars": "variable",
                "splits": "upper_bound_left",
                "left": "levels_left"
                }, axis=1)
            

            groves[[len(groves)]] = df
            interpretation[[len(interpretation)]] = df_small
            explanation = explanation.append(nt, len(df_small), upsilon, rho)

        # end of for every tree
        groves = pd.DataFrame(groves)
        interpretation = pd.DataFrame(interpretation)
        explanation = pd.DataFrame(explanation)

        groves.columns = self.ntrees
        interpretation.columns = self.ntrees
        explanation.columns = ["trees", "rules", "upsilon", "cor"]

        self.explanation = explanation
        self.rules = interpretation
        self.groves = groves
        self.model = self.surrGrove

        self.result = self.get_result()
    # end of calculateGrove()

        # TODO explanation und interpretation füllen 
        # TODO add functionality of plot
