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
from sklearn2pmml import PMMLPipeline, sklearn2pmml
from pypmml import Model
    

# read testing dataset
# data = read_csv(r'C:\Users\jjacq\xgrove\data\HousingData.csv')

# # create dataframe 
# df = pd.DataFrame(data)

# TODO: delete direct directory reference
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

class grove():
    # define xgrove-class with default values
    # TODO add type check
    # print("its upgraded")
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
        self.data = self.encodeCategorical(data)
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
    # Überprüfen, ob pfun None ist
        if self.pfun is None:
            # Dynamisches Entfernen des Zielattributs (surrTarName) aus den Daten
            # X = self.data.drop(columns=[self.surrTarName])
            # y = self.data[self.surrTarName]
            
            # Trainiere das Surrogatmodell mit den Daten
            # self.surrGrove.fit(X, y)
            
            # Mache Vorhersagen für die Zielvariable
            target = self.model.predict(data)
        else:
            # Verwende die angegebene predictive function, um das Ziel zu berechnen
            target = pfun(model=self.model, data=self.data)
        
        return target

    
    def getGBM(self):
        grove = GradientBoostingRegressor(n_estimators=max(self.ntrees),
                                          learning_rate=self.shrink,
                                          subsample=self.b_frac)
        grove.fit(self.data, self.surrTar)
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

    def load_pmml_model(pmml_path):
        """
        Lädt ein PMML-Modell und gibt das Modellobjekt zurück.

        Args:
            pmml_path (str): Der Dateipfad zur PMML-Datei.

        Returns:
            model (pypmml.Model): Das geladene Modellobjekt.
        """
        try:
            # Lade das PMML-Modell
            model = Model.load(pmml_path)
            return model
        except Exception as e:
            print(f"Fehler beim Laden des PMML-Modells: {e}")
            return None

    def export_to_pmml(self):
        print("Exportiere Modelle als PMML...")
        X = self.data.drop(columns=[self.surrTarName])
        
        # Speichere GBM Modell
        pipeline = PMMLPipeline([
            ("classifier", self.surrGrove)
        ])
        sklearn2pmml(pipeline, "models/gbm_model.pmml")
        
        # Speichere das RandomForest-Modell (oder anderes übergebenes Modell)
        model_pipeline = PMMLPipeline([
            ("classifier", self.model)
        ])
        sklearn2pmml(model_pipeline, "models/analyzed_model.pmml")
        
        print("Modelle erfolgreich als PMML exportiert.")
        
        # Speichere die Trainings- und Testdatensätze als CSV
        self.save_datasets()

    def save_datasets(self):
        # Speichere den Datensatz für das Training
        self.data.to_csv("data/training_data.csv", index=False)
        print("Trainingsdaten als CSV gespeichert: training_data.csv")

        # Speichere den Datensatz für das Testen (falls verfügbar)
        if hasattr(self, 'data_test'):
            self.data_test.to_csv("data/testing_data.csv", index=False)
            print("Testdaten als CSV gespeichert: testing_data.csv")
        
    def calculateGrove(self):
        explanation = []
        groves = []
        interpretation = []
        data = self.data
        
        # for every tree
        for nt in self.ntrees:
            # predictions generation
            predictions = self.surrGrove.staged_predict(data)
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
                feature_index = int(rules_df.iloc[i]['feature'])
                # print("feature_index: ", feature_index)
                var_name = data.columns[int(feature_index)]
                vars.append(var_name)
                # print("isinstance(var_name, str): ", isinstance(var_name, str))
                # # Categorical columns
                
######################### Potentielle Fehlerquelle ####################################

                if pd.api.types.is_string_dtype(data.iloc[:,feature_index]) or isinstance(data.iloc[:,feature_index], str) or isinstance(data.iloc[:,feature_index], object):
                    #print(i+": Kategorisch")
                    levs = data[var_name].unique()
                    lids = self.surrGrove.estimators_[0, 0].tree_.value[int(rules_df.iloc[i]['threshold'])] == -1
                    if sum(lids) == 1: levs = levs[lids]
                    if sum(lids) > 1: levs = " | ".join(levs[lids])
                    csl = levs[0] if isinstance(levs, (list, pd.Index)) else levs
                    if len(levs) > 1:
                        csl = " | ".join(str(levs))

                    splits.append("")
                    csplits_left.append(csl)
                
                elif isinstance(data.iloc[:,i], pd.Categorical):
                    levs = rules_df.columns[i].cat.categories
                    lids = self.surrGrove.estimators_[0, 0].tree_.value[int(rules_df.iloc[i]['threshold'])] == -1
                    if sum(lids) == 1: levs = levs[lids]
                    if sum(lids) > 1: levs = " | ".join(levs[lids])
                    csl = levs[0] if isinstance(levs, (list, pd.Index)) else levs
                    if len(levs) > 1:
                        csl = " | ".join(levs)

                    splits.append("")
                    csplits_left.append(csl)

                # Numeric columns   
                elif pd.api.types.is_numeric_dtype(data.iloc[:,i]) or np.issubdtype(data.iloc[:,i], np.number):
                    #print(i+": Numerisch")
                    splits = splits.append(rules_df.iloc[i]["threshold"])
                    csplits_left.append(pd.NA)

                else:
                    print(rules_df[i]+": uncaught case")
            # # rules filled
            # print("i: ", i)
            # print("Länge rules_df: ", len(rules_df))

            pleft.append(rules_df.loc[:,"pleft"])
            pright.append(rules_df.loc[:,"pleft"])

            # # print("pright.len: ",len(np.array(round(elem, 4) for elem in pright)))
            # print()
            # print("vars.len: ",len(vars))
            # print("splits.len: ",len(splits))

            pleft = np.array(round(elem, 4) for elem in pleft)
            pright = np.array(round(elem, 4) for elem in pright)

            basepred = self.surrGrove.estimators_
            
            df = pd.DataFrame({
                "vars": vars,
                "splits": splits,
                "left": csplits_left,
                "pleft": pleft,
                "pright": pright
            })
            # print(df)
            # print("vars: ", df.loc[:,"vars"])
            # print("splits: ", df.loc[:,"splits"])
            # print("left: ", df.loc[:,"left"])

            df_small = df.groupby(["vars", "splits", "left"], as_index=False).agg({"pleft" : "sum", "pright" : "sum"})
            # df_small.set_index(["vars", "splits", "left"], inplace=True)
            # df_small.index.set_names(["vars", "splits", "left"], inplace=True)
            # print(df_small)
            # print(df_small.shape)
            # print(df_small.columns)
            # print(df_small.index.names)

            if(len(df_small) > 1):
                i = 2
                while (i != 0):
                    drop_rule = False
                    # check if its numeric AND NOT categorical
                    # all_vars = df_small.index.get_level_values('vars')

                    # print("all_vars: ",all_vars)
                    # print("df_small: ",df_small)
                    # print(i)
                    # print("vars at ",i,": ", df_small["vars"].iloc[i])

                    if pd.api.types.is_numeric_dtype(self.data[df_small["vars"].iloc[i]])or np.issubdtype(self.data[df_small["vars"].iloc[i]], np.number) and not(isinstance(self.data[df_small["vars"].iloc[i]], pd.Categorical | object | str) or pd.api.types.is_string_dtype(self.data[df_small["vars"].iloc[i]])):
                        #print(i+": Numerisch")
                        for j in range(0, i):
                            if df_small["vars"][i] == df_small["vars"][j]:
                                v1 = self.data[df_small["vars"][i]] <= df_small["splits"][i]
                                v2 = data[df_small["vars"][j]] <= df_small["splits"][j]
                                tab = [v1,v2]
                                if tab.values.trace() == tab.values.sum():
                                    df_small.at[j, 'pleft'] = df_small.at[i, 'pleft'] + df_small.at[j, 'pleft']
                                    df_small.at[j, 'pright'] = df_small.at[i, 'pright'] + df_small.at[j, 'pright']
                                    drop_rule = True
                    if drop_rule: df_small = df_small[-i,]
                    if not drop_rule: i = i+1
                    if i+1 > len(df_small): i = 0
            # compute complexity and explainability statistics
            upsilon, rho = self.upsilon(pexp=predictions)

            df0 = pd.DataFrame({
                "vars": ["Interept"],
                "splits": [pd.NA],
                "left": [pd.NA],
                "pleft": [basepred],
                "pright": [basepred]
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
            

            groves[len(groves)-1] = df
            interpretation.append(df_small)
            explanation.append({
                "trees": nt,
                "rules":len(df_small),
                "upsilon":upsilon,
                "cor": rho
                })

        # end of for every tree
        # groves = pd.DataFrame(groves)
        # interpretation = pd.DataFrame(interpretation)
        explanation = pd.DataFrame(explanation)

        # groves.index = self.ntrees
        # interpretation.index = self.ntrees
        # # explanation.columns = ["trees", "rules", "upsilon", "cor"]

        self.explanation = explanation
        self.rules = interpretation
        self.groves = groves
        self.model = self.surrGrove

        self.result = self.get_result()
    # end of calculateGrove()

        # TODO explanation und interpretation füllen 
        # TODO add functionality of plot

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# 1. Erstellen eines synthetischen Datensatzes
# erklären wie die daten erstellt werden
# erklären was dieses Modell genau für ein modell ist
X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
# data['target'] = y

# 2. Aufteilen der Daten in Trainings- und Testdaten
# data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)

# 3. Initialisieren des RandomForestRegressors
rf_Model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_Model.fit(X,y)

# 4. Instanziieren der Grove-Klasse
grove_model = grove(data=data, model=rf_Model)

# 5. Berechnen des Groves
results = grove_model.calculateGrove()

# 6. Ergebnisse anzeigen
print("Berechnungen abgeschlossen.")
print(grove_model.result)

