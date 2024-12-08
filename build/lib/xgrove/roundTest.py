import numpy as np
from scipy.stats import pearsonr

# Simulierte Testdaten
surrTar = np.array([3, 2, 7, 8, 1])  # tatsächliche Zielwerte
pexp = np.array([2.8, 2.1, 6.9, 8.1, 1.2])  # Vorhersagen des Modells

# Berechnung von ASE und ASE0
ASE = np.mean((surrTar - pexp) ** 2)
ASE0 = np.mean((surrTar - np.mean(surrTar)) ** 2)

# Berechnung von Upsilon
upsilon = 1 - ASE / ASE0

# Korrelation
# correlation = np.corrcoef(surrTar, pexp)[0, 1]
correlation, _ = pearsonr(surrTar, pexp)

print(f"ASE: {ASE}")
print(f"ASE0: {ASE0}")
print(f"Upsilon: {upsilon}")
print(f"Correlation: {correlation}")