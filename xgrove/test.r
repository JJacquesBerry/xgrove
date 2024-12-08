# Benötigte Bibliotheken laden
library(randomForest)
library(pmml)
library(XML)
library(xgrove)

# Beispiel-Daten erstellen
set.seed(42)
# data <- data.frame(
#   Feature1 = rnorm(100),
#   Feature2 = rnorm(100),
#   Target = rnorm(100)  # Zielvariable als kontinuierliche Variable (für Regression)
# )

data <- data.frame(
  Feature1 = sample(0:100, 100, replace = TRUE),  # Ganzzahlige Werte zwischen 0 und 100
  Feature2 = sample(0:100, 100, replace = TRUE),  # Ganzzahlige Werte zwischen 0 und 100
  Target = sample(0:100, 100, replace = TRUE)     # Zielvariable mit ganzzahligen Werten
)

# Trainiere das lineare Regressionsmodell
lm_model <- lm(Target ~ Feature1 + Feature2, data = data)

# Speichere das Modell als PMML-Datei
pmml_path <- "models/linear_model.pmml"  # Originaler Pfad
pmml_model <- pmml(lm_model)
saveXML(pmml_model, file = pmml_path)

# Speichern des Datensatzes als CSV mit maximaler Präzision
data_path <- "models/generated_data.csv"  # Originaler Pfad

# # Verwende format() für die Daten, um die Präzision zu erhöhen, aber nicht den Datentyp zu verändern
# data$Feature1 <- format(data$Feature1, digits = 16)
# data$Feature2 <- format(data$Feature2, digits = 16)
# data$Target <- format(data$Target, digits = 16)

# # Konvertiere die Spalten zurück zu numerischen Werten (nur nach der Formatierung)
# data$Feature1 <- as.numeric(data$Feature1)
# data$Feature2 <- as.numeric(data$Feature2)
# data$Target <- as.numeric(data$Target)

# Speichere den Datensatz ohne Zeilenummern und ohne Anführungszeichen für Zahlen
write.csv(data, file = data_path, row.names = FALSE, quote = FALSE)

cat("Der Datensatz wurde unter 'models/generated_data.csv' mit maximaler Präzision gespeichert.\n")

# Schritt 2: Surrogat-Grove erstellen
data_without_target <- data[, !names(data) %in% "Target"]
cat("Der Datensatz wurde unter 'models/generated_data.csv' gespeichert.\n")
print(head(data))

# Anzahl der Bäume, die für das Grove-Modell getestet werden
ntrees <- c(4, 8, 16, 32, 64, 128)

# Verwende xgrove, um das Surrogat-Grove-Modell zu erstellen
surrogate_grove <- xgrove(
  model = lm_model,  # Das lineare Modell als Basis für das Surrogat
  data = data_without_target,
  ntrees = ntrees,
  shrink = 1,
  b.frac = 1,
  seed = 42
)

# Vorhersage des linearen Modells auf data_without_target
predicted_tar_lm <- predict(lm_model, newdata = data_without_target)

# Kombiniere die Vorhersagen in einem DataFrame
predictions <- data.frame(
  predicted_tar_lm = predicted_tar_lm
)

# Speichern der Vorhersagen als CSV mit maximaler Präzision
write.csv(predictions, file = "models/predictions.csv", row.names = FALSE, quote = FALSE)

# Bestätigung der Speicherung
cat("Die Vorhersagen wurden unter 'predictions.csv' mit maximaler Präzision gespeichert.\n")

# Anzeige der Surrogatmodell-Ergebnisse
print(surrogate_grove)

# Optional: Plot zur Analyse der Komplexität des Grove-Modells im Vergleich zur Erklärbarkeit
plot(surrogate_grove, abs = "trees", ord = "upsilon")

# Bestätigung der Speicherung des Datensatzes
cat("Der Datensatz wurde unter 'models/generated_data.csv' gespeichert.\n")

# Speichern der Skalierungsparameter (Mittelwerte und Standardabweichungen)
train_data <- data_without_target
train_data_scaled <- scale(train_data)

means <- attr(train_data_scaled, "scaled:center")
sds <- attr(train_data_scaled, "scaled:scale")
print(means)
print(sds)


# Simulierte Testdaten
surrTar <- c(3, 2, 7, 8, 1)  # tatsächliche Zielwerte
pexp <- c(2.8, 2.1, 6.9, 8.1, 1.2)  # Vorhersagen des Modells

# Berechnung von ASE und ASE0
ASE <- mean((surrTar - pexp)^2)
ASE0 <- mean((surrTar - mean(surrTar))^2)

# Berechnung von Upsilon
upsilon <- 1 - ASE / ASE0

# Korrelation
correlation <- cor(surrTar, pexp)

cat("ASE:", ASE, "\n")
cat("ASE0:", ASE0, "\n")
cat("Upsilon:", upsilon, "\n")
cat("Correlation:", correlation, "\n")

