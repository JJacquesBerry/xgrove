# Benötigte Bibliotheken importieren
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import xgrove as xg

# Schritt 1: CSV-Datei einlesen (Daten aus der bestehenden Datei)
data_path = "../models/generated_data.csv"
data = pd.read_csv(data_path)

# Anzeige der ersten Zeilen, um sicherzustellen, dass die Daten korrekt geladen wurden
print("Daten geladen:")
print(data.head())

# Schritt 2: Lineares Regressionsmodell erstellen
X = data[['Feature1', 'Feature2']]  # Unabhängige Variablen
y = data['Target']  # Zielvariable

# Modell erstellen und trainieren
lm_model = LinearRegression()
lm_model.fit(X, y)

# Anzeige der Koeffizienten und des Intercepts
# print(f"Koeffizienten: {lm_model.coef_}")
# print(f"Intercept: {lm_model.intercept_}")

# Schritt 3: Vorhersage mit dem Modell machen
predictions_lm_new = lm_model.predict(X)

# Die bestehende CSV-Datei einlesen
predictions_df = pd.read_csv('../models/predictions.csv', dtype={'predicted_tar_lm': np.float64})

# Vorhersagen für lm_model als neue Spalte hinzufügen
predictions_df['predicted_tar_py'] = predictions_lm_new

# Die aktualisierte Datei speichern (anfügen und nicht überschreiben)
predictions_df.to_csv('../models/predictions.csv', index=False)  # Maximale Genauigkeit von 16 Dezimalstellen

# Bestätigung der Speicherung
print("Die Vorhersagen wurden erfolgreich zu 'predictions.csv' hinzugefügt.")

# Schritt 4: Berechnung der durchschnittlichen Abweichung
# abweichung = np.mean(np.abs(y - predictions_lm_new))
# print(f"Durchschnittliche Abweichung: {abweichung}")

grove = xg.grove(data = data.drop("Target", axis=1), seed=42, model=lm_model)
grove.calculateGrove()
grove.get_result()
