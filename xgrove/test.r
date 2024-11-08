# Benötigte Bibliotheken laden
library(randomForest)
library(pmml)
library(XML)
library(xgrove)

# Zufallsgenerator für die Reproduzierbarkeit
set.seed(123)

# Beispiel-Daten erstellen
x <- rnorm(500, mean = 50, sd = 10)  # Zufällige unabhängige Variable
y <- 2.5 * x + rnorm(500, mean = 0, sd = 5)  # Abhängige Variable mit einer linearen Beziehung zu x

# Erstelle einen DataFrame für die Daten
data <- data.frame(x = x, y = y)

# Speichern der Trainingsdaten mit nur der unabhängigen Variablen 'x' in einer CSV-Datei
data_no_target <- data[, !names(data) %in% "y"]  # Entferne die Zielvariable 'y'
write.csv(data_no_target, "models/generated_data.csv", row.names = FALSE)  # Nur x wird gespeichert

# Random Forest Modell anstelle eines lm-Modells
# So behältst du die lineare Struktur und kannst xgrove nutzen
modell_rf <- randomForest(y ~ x, data = data, ntree = 100, maxnodes = 2)  # maxnodes gering halten, um lineare Beziehungen zu simulieren

# Modellzusammenfassung
print(modell_rf)

# Verwende xgrove mit dem Random Forest Modell und dem DataFrame
tryCatch({
  xg <- xgrove(modell_rf, data)
  plot(xg)
}, error = function(e) {
  cat("Fehler bei der Anwendung von xgrove:", e$message, "\n")
})

# PMML-Dateipfad definieren
pmml_file_path <- "models/linear_model.pmml"

# Funktion zum Speichern des Random Forest Modells als PMML
save_model_as_pmml <- function(model, file_path) {
  pmml_model <- pmml(model)  # PMML-Modell erstellen
  
  if (is.null(pmml_model)) {
    stop("Fehler beim Erstellen des PMML-Modells.")
  }
  
  saveXML(pmml_model, file = file_path)  # PMML-Modell in Datei speichern
}

# Speichern des Modells als PMML
tryCatch({
  save_model_as_pmml(modell_rf, pmml_file_path)
  print("Random Forest Modell als PMML gespeichert.")
}, error = function(e) {
  cat("Fehler beim Speichern des Modells:", e$message, "\n")
})
