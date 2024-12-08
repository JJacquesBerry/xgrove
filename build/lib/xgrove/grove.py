import mkdocs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import pearsonr

class grove():
    def __init__(self, 
                 model, 
                 data: pd.DataFrame,
                 ntrees: np.array = np.array([4, 8, 16, 32, 64, 128]), 
                 pfun=None, 
                 shrink: int = 1, 
                 b_frac: int = 1, 
                 seed: int = 42,
                 tar=None):
        self.model = model
        self.data = self.encodeCategorical(data)
        self.ntrees = ntrees
        self.pfun = pfun
        self.shrink = shrink
        self.b_frac = b_frac
        self.seed = seed
        self.tar = tar
        self.surrTar = self.getSurrogateTarget(pfun=self.pfun, tar=self.tar)
        self.surrGrove = self.getGBM()
        self.explanation = []
        self.groves = []
        self.rules = []
        self.result = []

    def encodeCategorical(self, data):
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        return pd.get_dummies(data, columns=categorical_columns)

    def getSurrogateTarget(self, pfun, tar):
        if tar is None:
            return self.model.predict(self.data) if self.pfun is None else pfun(model=self.model, data=self.data)
        else:
            return tar

    def getGBM(self):
        grove = GradientBoostingRegressor(max_depth=1, n_estimators=max(self.ntrees), learning_rate=self.shrink, subsample=self.b_frac, random_state=self.seed, criterion='squared_error', min_samples_leaf=10)
        grove.fit(self.data, self.surrTar)
        return grove
    
    def upsilon(self, pexp):
        surrTar_series = pd.Series(self.surrTar)
        pexp_series = pd.Series(pexp)
        ASE = np.mean((surrTar_series - pexp_series) ** 2)
        ASE0 = np.mean((surrTar_series - np.mean(surrTar_series)) ** 2)
        ups = np.round(1 - ASE / ASE0, 7)
        rho = np.round(pearsonr(surrTar_series, pexp_series), 7)
        return ups, rho

    def get_result(self):
        return [self.explanation, self.rules, self.groves, self.model]

    def consolidate_rules(self, df_small):
        # Für jede Regel (ab der zweiten)
        data = self.data
        i = 1
        while i < len(df_small):
            drop_rule = False
            # Überprüfe, ob die Variable numerisch ist
            if pd.api.types.is_numeric_dtype(data[df_small.iloc[i]["variable"]]):
                # Vergleiche jede Regel mit den vorherigen Regeln
                for j in range(i):
                    if df_small.iloc[i]["variable"] == df_small.iloc[j]["variable"]:
                        # Check if the upper_bound_left values are the same
                        if df_small.iloc[i]["upper_bound_left"] == df_small.iloc[j]["upper_bound_left"]:
                            v1 = data[df_small.iloc[i]["variable"]] <= df_small.iloc[i]["upper_bound_left"]
                            v2 = data[df_small.iloc[j]["variable"]] <= df_small.iloc[j]["upper_bound_left"]
                            tab = pd.crosstab(v1, v2)
                            
                            # Wenn die Bedingung für beide Variablen übereinstimmt
                            if tab.to_numpy().diagonal().sum() == tab.sum().sum():
                                # Fasse pleft und pright zusammen
                                df_small.at[j, "pleft"] += df_small.at[i, "pleft"]
                                df_small.at[j, "pright"] += df_small.at[i, "pright"]
                                drop_rule = True
                                break
            # Lösche die Regel, wenn sie redundant ist
            if drop_rule:
                df_small = df_small.drop(i).reset_index(drop=True)
            else:
                i += 1
            if i >= len(df_small):
                break

        return df_small

    def plot_xgrove(self, abs_col="trees", ord_col="upsilon", **kwargs):
        # Hole die Indizes der entsprechenden Spalten
        abs_index = self.explanation.columns.get_loc(abs_col)
        ord_index = self.explanation.columns.get_loc(ord_col)
        
        # Extrahiere die Werte für das Plotten
        x_values = self.explanation.iloc[:, abs_index]
        y_values = self.explanation.iloc[:, ord_index]
        
        # Erstelle das Plot
        plt.plot(x_values, y_values, marker='o', linestyle='-', **kwargs)
        plt.xlabel(abs_col)
        plt.ylabel(ord_col)
        plt.title(f'{abs_col} vs. {ord_col}')
        plt.grid(True)
        plt.show()


    def calculateGrove(self):
        explanation = []
        cumulative_rules_list = []  # List to store cumulative rules per tree count
        data = self.data

        # Start with an empty DataFrame for cumulative rules

        # Generate predictions for all trees and accumulate rules as required
        predictions_gen = self.surrGrove.staged_predict(data)

        # For every tree count (nt)
        for nt in self.ntrees:
            cumulative_rules = pd.DataFrame()

            # Set predictions for the current grove with `nt` trees
            for i, prediction in enumerate(predictions_gen, start=1):
                if i == nt:
                    predictions = prediction  # Setze die Vorhersagen für den Zustand bis `nt` Bäume
                    break
            rules = []  # List to store rules for the current grove

            # Extract rules from each tree in the current grove
            for tid in range(nt):
                tree = self.surrGrove.estimators_[tid, 0].tree_

                # Process each node in the tree to extract split information
                for node_id in range(tree.node_count):
                    if tree.children_left[node_id] != tree.children_right[node_id]:  # Split node
                        rule = {
                            'feature': data.columns[tree.feature[node_id]],  # Klarname der Variablen
                            'threshold': tree.threshold[node_id],
                            'pleft': round(tree.value[tree.children_left[node_id]][0][0], 4),
                            'pright': round(tree.value[tree.children_right[node_id]][0][0], 4)
                        }
                        rules.append(rule)

            # Create DataFrame for current grove's rules
            rules_df = pd.DataFrame(rules)

            # Add current rules to cumulative rules
            cumulative_rules = pd.concat([cumulative_rules, rules_df], ignore_index=True)
            cumulative_rules_list.append(cumulative_rules.copy())

            # Compute upsilon and rho
            upsilon, rho = self.upsilon(pexp=predictions)

            
            # Prepare interpretation of current cumulative rules
            vars_temp = []
            splits_temp = []
            csplits_left_temp = []
            pleft_temp = []
            pright_temp = []

            for i in range(len(cumulative_rules)):
                var_name = cumulative_rules.iloc[i]['feature']  # Klarname, ohne feature_index
                threshold = cumulative_rules.iloc[i]["threshold"]
                pleft = cumulative_rules.iloc[i]["pleft"]
                pright = cumulative_rules.iloc[i]["pright"]

                vars_temp.append(var_name)
                if pd.api.types.is_string_dtype(data[var_name]):
                    levels = data[var_name].unique()
                    csplits_left_temp.append(" | ".join(map(str, levels)))
                    splits_temp.append("")
                else:
                    splits_temp.append(threshold)
                    csplits_left_temp.append(pd.NA)
                pleft_temp.append(pleft)
                pright_temp.append(pright)

            # Construct interpretation DataFrame
            df = pd.DataFrame({
                "variable": vars_temp,
                "upper_bound_left": splits_temp,
                "levels_left": csplits_left_temp,
                "pleft": pleft_temp,
                "pright": pright_temp
            })

            # Add Intercept
            intercept_df = pd.DataFrame({
                "variable": ["Intercept"],
                "upper_bound_left": [pd.NA],
                "levels_left": [pd.NA],
                "pleft": [self.surrGrove.estimators_[0, 0].tree_.value[0][0]],
                "pright": [self.surrGrove.estimators_[0, 0].tree_.value[0][0]]
            })
            intercept_df = intercept_df.fillna('default')
            # Debugging: Check DataFrame before grouping
            # print(f"DataFrame before grouping: {df}")

            df['levels_left'] = df['levels_left'].fillna('default')
            # 1. Entferne Zeilen mit NaN-Werten in den relevanten Spalten oder ersetze NaN durch einen Platzhalter
            df = df.dropna(subset=["upper_bound_left", "levels_left"], how="any")

            # Debugging: Check NaN counts after removal
            # print(f"NaN counts after handling: {df.isnull().sum()}")

            # 2. Gruppierung durchführen
            df_small = df.groupby(["variable", "upper_bound_left", "levels_left"], as_index=False).agg({
                "pleft": "sum",
                "pright": "sum"
            })

            # Debugging: Check df_small after grouping
            print(f"df_small grouped: {df_small}")

            # Add intercept to the main df and the grouped df_small
            df = pd.concat([intercept_df, df], ignore_index=True)
            df_small = pd.concat([intercept_df, df_small], ignore_index=True)

            # Debugging: Final check on df_small after concatenation
            # print(f"df_small after concatenation: {df_small}")

            # Prepare explanations
            explanation.append({
                "trees": nt,  
                "rules": len(df_small)-1, # intercept abgezogen 
                "upsilon": upsilon,
                "cor": rho[0]
            })

            # Store the cumulative interpretation
            self.rules.append(df_small)

        # Store explanations, cumulative rules, and groves
        self.explanation = pd.DataFrame(explanation)
        self.groves = cumulative_rules_list
        self.result = self.get_result()