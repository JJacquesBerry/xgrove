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
                 grove_rate: float = 1,
                 ):
        self.model = model
        self.data = self.encodeCategorical()
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

# get-functions for class overarcing variables
    def getSurrogateTarget(self, pfun):
        if(self.pfun == None):
            target = self.model.predict(self.data)
        else:
            target = pfun(model = self.model, data = self.data)
        return target
    
    def getGBM(self):

        grove = GradientBoostingRegressor(n_estimators=self.ntrees,
        learning_rate=self.shrink,
        subsample=self.b_frac)
        return grove
    # OHE for evaluating categorical columns
    def encodeCategorical(self):
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns
        data_encoded = pd.get_dummies(data, columns=categorical_columns)
        return data_encoded

    # calculate upsilon
    # pexp = viewed predictive model
    def upsilon(self, pexp):
        ASE  = statistics.mean((self.surrTar-pexp)**2)
        ASE0 = statistics.mean((self.surrTar-statistics.mean(self.surrTar))**2)
        ups = 1 - ASE/ASE0
        rho = statistics.correlation(self.surrTar, pexp)
        return ups, rho
    
    def get_result(self):
        res = [self.explanation, self.rules, self.groves, self.model]
        return res
    
# TODO define plot method for the Upsilon-Rules-Curve of the surrogate grove
    def plot(x,
             abs = "rules",
             ord = "upsilon"):
        abs = abs
        ord = ord
        return plt.plot()
    
    # compute performance and extract groves
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
        return(self.result)
    # end of calculateGrove()

        # TODO explanation und interpretation füllen 
        # TODO add functionality of plot
class sgtree():
    def __init__(self, 
                 model, 
                 data: pd.DataFrame, 
                 maxdeps: np.array = np.array(range(1,9)), 
                 cparam = 0,
                 pfun = None
                 ):
        self.model = model
        self.data = self.encodeCategorical()
        self.maxdeps = maxdeps
        self.cparam = cparam
        self.pfun = pfun
        self.surrTar = self.getSurrogateTarget(pfun)
        self.surrogate_trees = []
        self.rules  = []
        self.explanation = []   

    def encodeCategorical(self):
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns
        data_encoded = pd.get_dummies(data, columns=categorical_columns)
        return data_encoded
    
    # compute surrogate grove for specified maximal number of trees
    def getSurrogateTarget(self, pfun):
        if(self.pfun == None):
            self.surrTar = self.model.predict(self.data)
        else:
            self.surrTar = pfun(model = self.model, data = self.data)
        
    def calcusatesgtree(self):
        surrogate_trees = []
        surrogate_trees.index = str(self.maxdeps)
        explanation = pd.DataFrame
        explanation.columns = ["trees","rules","upsilon","cor"]
        cat_col = []
        num_col = []
        for i in self.data.columns:    
            if i.dtype == pd.Categorical or pd.api.types.is_string_dtype(i) or i.dtype == object:
                cat_col.append(i)
            # Numeric columns   
            elif pd.api.types.is_numeric_dtype(i) or np.issubdtype(i.dtype, np.number):
                num_col.append(i)
            else:
                print(i+": uncaught case please contact a dev")
        
        for md in self.maxdeps:
    # min_samples_split = minsplit, min_samples_leaves = minbucket?
            model = tree.DecisionTreeRegressor(max_depth=md, ccp_alpha=self.cparam, min_samples_split = 2, min_samples_leaf = 1).fit(X=self.data, y=self.surrTar)
            t = model.tree_
            features = t.feature
            thresholds = t.threshold
            rules = []
            csplits_left = []
            
            for node in range(t.node_count):
                ncat = 
                # if it's not a leaf node save attributes to list object "rules"    
                if features[node].dtype == pd.Categorical or pd.api.types.is_string_dtype(features[node]) or features[node].dtype == object:
                    ncat.append(datafeatures[node])
                # Numeric columns   
                elif pd.api.types.is_numeric_dtype(features[node]) or np.issubdtype(features[node].dtype, np.number):
                    ncat.append(-1)
                else:
                    print(i+": uncaught case please contact a dev")
            
                if features[node] != -2:
                    rule = {
                        # feature == var
                        'feature': features[node],
                        'threshold': thresholds[node],
                        'pleft': t.value[t.children_left[node]][0][0],
                        'pright': t.value[t.children_right[node]][0][0],
                        'ncat': ncat
                    }
                    rules.append(rule)
            rules_df = pd.DataFrame(rules)
            
            # surrogate_trees.append(pd.DataFrame(rules))

            if len(surrogate_trees[md]) == 0:
                explanation.append(pd.DataFrame({
                        "trees": 1,
                        "rules": 0,
                        "upsilon": 0,
                        "cor": 0
                }))
                surrogate_trees[md] = None

            if len(surrogate_trees[md] > 0):
                    # if t.node_count>1:

                    md = md
            
        print(surrogate_trees)
        # TODO: splits äquivalent finden
    
    
    
    

    
