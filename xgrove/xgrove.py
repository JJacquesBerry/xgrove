import sklearn
import sklearn.datasets
import sklearn.metrics as metrics
import sklearn.model_selection
import sklearn.tree as tree
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification
from pandas import read_csv

# read testing dataset
data = read_csv(r'C:\Users\jjacq\xgrove\data\HousingData.csv')

# create dataframe 
df = pd.DataFrame(data)
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

from sklearn import preprocessing
import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

encoder = preprocessing.OneHotEncoder(sparse=None)
df_enc = encoder.fit_transform(df[categorical_columns])

y = pd.DataFrame(data['MEDV'])
df = df.drop(['MEDV'], axis=1)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df, y, test_size=0.2, random_state=42)
t = tree.DecisionTreeRegressor(random_state=42, max_depth=4)

rf_Model = RandomForestRegressor(random_state=42, max_depth=4,max_samples=404)
rf_Model.fit(X_train,y_train)

print(X_train)
print(X_test)
print(y_train)
print(y_test)

t = t.fit(X_train, y_train)
y_pred = t.predict(X_test)
mse = metrics.mean_squared_error(y_test, y_pred)

# generate a graph of the generated tree/grove
tree_graph = tree.export_graphviz(t, out_file=None, filled=True, rounded=True, feature_names=df.columns)
graph = graphviz.Source(tree_graph, format='png')
graph.render('decision_tree_graphviz')

#define etimated squared difference equation
import scipy.integrate as integrate

# calculate esd of a prediction model as a number between 0 and 1
def estimated_squared_difference(f):
    integrand = lambda x: f - x
    integral, integral_error = integrate.quad(integrand, 0, 1)
    return (integral,integral_error)

# calculate upsilon using esd from above
import statistics
def upsilon(porig, pexp):
    ASE  = statistics.mean((porig-pexp)**2)
    ASE0 = statistics.mean((porig-statistics.mean(porig))**2)
    ups = 1- ASE/ASE0
    return ups
class xgrove():
    # define xgrove-class with default values
    def __init__(self, model, data, 
                 ntrees = np.array([4, 8, 16, 32, 64, 128]), 
                 pfun = None, 
                 shrink = 1, 
                 b_frac = 1, 
                 seed = 42):
        self.model = model,
        self.data = data,
        self.ntrees = ntrees,
        self.pfun = pfun,
        self.shrink = shrink,
        self.b_frac = b_frac,
        self.seed = seed
    # define plot method for the Upsilon-Rules-Curve of the surrogate grove
    def plot(x,
             abs = "rules",
             ord = "upsilon"):
        abs = abs
        ord = ord
        plt.plot()
    




xg = xgrove(model=rf_reg, data=boston)
print(xg)
# def xgrove(
#         model,
#         data,
#         ntrees = np.array([4, 8, 16, 32, 64, 128]),
#         pfun = None,
#         shrink = 1,
#         b_frac = 1,
#         seed = 42):
    





# ###TESTAREA###
# sklearn.tree
# import numpy as np
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import accuracy_score, classification_report

# # Laden des Iris-Datensets
# iris = load_iris()
# X = iris.data
# y = iris.target

# # Aufteilen der Daten in Trainings- und Testsets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Initialisieren und Trainieren des Gradient Boosting Classifiers
# gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)
# gb_clf.fit(X_train, y_train)

# # Vorhersagen auf dem Testset
# y_pred = gb_clf.predict(X_test)

# # Berechnen der Genauigkeit
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Genauigkeit: {accuracy * 100:.2f}%")

# # Ausgabe eines detaillierten Klassifikationsberichts
# print(classification_report(y_test, y_pred, target_names=iris.target_names)