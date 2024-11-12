# # Benötigte Bibliotheken laden
# library(randomForest)
# library(pmml)
# library(XML)
# library(xgrove)

# # Zufallsgenerator für die Reproduzierbarkeit
# set.seed(123)

# # Beispiel-Daten erstellen
# x <- rnorm(500, mean = 50, sd = 10)  # Zufällige unabhängige Variable
# y <- 2.5 * x + rnorm(500, mean = 0, sd = 5)  # Abhängige Variable mit einer linearen Beziehung zu x

# # Erstelle einen DataFrame für die Daten
# data <- data.frame(x = x, y = y)

# # Speichern der Trainingsdaten mit nur der unabhängigen Variablen 'x' in einer CSV-Datei
# data_no_target <- data[, !names(data) %in% "y"]  # Entferne die Zielvariable 'y'
# write.csv(data_no_target, "models/generated_data.csv", row.names = FALSE)  # Nur x wird gespeichert

# # Random Forest Modell anstelle eines lm-Modells
# # So behältst du die lineare Struktur und kannst xgrove nutzen
# modell_rf <- randomForest(y ~ x, data = data, ntree = 100, maxnodes = 2)  # maxnodes gering halten, um lineare Beziehungen zu simulieren

# # Modellzusammenfassung
# print(modell_rf)

# # Verwende xgrove mit dem Random Forest Modell und dem DataFrame
# tryCatch({
#   xg <- xgrove(modell_rf, data)
#   plot(xg)
# }, error = function(e) {
#   cat("Fehler bei der Anwendung von xgrove:", e$message, "\n")
# })

# # PMML-Dateipfad definieren
# pmml_file_path <- "models/linear_model.pmml"

# # Funktion zum Speichern des Random Forest Modells als PMML
# save_model_as_pmml <- function(model, file_path) {
#   pmml_model <- pmml(model)  # PMML-Modell erstellen
  
#   if (is.null(pmml_model)) {
#     stop("Fehler beim Erstellen des PMML-Modells.")
#   }
  
#   saveXML(pmml_model, file = file_path)  # PMML-Modell in Datei speichern
# }

# # Speichern des Modells als PMML
# tryCatch({
#   save_model_as_pmml(modell_rf, pmml_file_path)
#   print("Random Forest Modell als PMML gespeichert.")
# }, error = function(e) {
#   cat("Fehler beim Speichern des Modells:", e$message, "\n")
# })
# In R
# Erstelle die Beispiel-Daten
# Benötigte Bibliotheken laden

# Benötigte Bibliotheken laden# Benötigte Bibliotheken laden
# library(randomForest)
# library(pmml)
# library(xml2)

# # Beispiel: Zufällige Daten erstellen
# set.seed(123)
# data <- data.frame(x = rnorm(500, mean = 50, sd = 10), 
#                    y = sample(c(0, 1), 500, replace = TRUE))  # Binäre Zielvariable

# # Sicherstellen, dass 'y' als Faktor behandelt wird (für Klassifikation)
# data$y <- as.factor(data$y)

# # Trainiere ein Random Forest Modell als Klassifikation
# model_rf <- randomForest(y ~ x, data = data, ntree = 100)

# # Modell zusammenfassen
# print(model_rf)

# # PMML-Dateipfad definieren
# pmml_file_path <- "models/random_forest_model_pmml.xml"

# # Sicherstellen, dass der Ordner 'models' existiert
# dir.create("models", showWarnings = FALSE)

# # PMML-Modell mit pmml erstellen und speichern
# pmml_obj <- pmml(model_rf)

# # Speichern des PMML-Modells als XML-Datei
# writeLines(toString(pmml_obj), con = pmml_file_path)

# # Bestätigung der Speicherung
# cat("PMML-Modell gespeichert unter:", pmml_file_path, "\n")

# # PMML-Datei laden
# pmml_data <- read_xml(pmml_file_path)

# # Modelltyp anzeigen (sollte 'classification' sein)
# model_type <- xml_find_first(pmml_data, "//pmml//MiningModel")
# cat("Modelltyp: ", xml_attr(model_type, "functionName"), "\n")

# # Eingabefelder anzeigen
# input_fields <- xml_find_all(pmml_data, "//pmml//DataDictionary//DataField")
# cat("Eingabefelder: \n")
# cat(paste(xml_attr(input_fields, "name"), collapse = ", "), "\n")

# # Zielvariable anzeigen
# target_field <- xml_find_first(pmml_data, "//pmml//MiningSchema//MiningField[@usageType='target']")
# cat("Zielvariable: ", xml_attr(target_field, "name"), "\n")

# # Lade benötigte Pakete
# library(dplyr)
# library(caret)
# library(pmml)

# # Erstelle ein simples DataFrame mit maximal 100 Einträgen
# set.seed(123)
# data <- data.frame(
#   Feature1 = rnorm(100),
#   Feature2 = rnorm(100),
#   Target = rnorm(100)  # Zielvariable als kontinuierliche Variable (für Regression)
# )

# # Trainiere ein lineares Regressionsmodell
# model <- train(Target ~ Feature1 + Feature2, data = data, method = "lm")

# # Speichere das Modell als PMML-Datei
# pmml_model <- pmml(model$finalModel)
# saveXML(pmml_model, file = "models/linear_model.pmml")


# Lade benötigte Pakete
library(dplyr)
library(caret)
library(pmml)
library(gbm)
library(xgrove)

# Step 1: Lineares Regressionsmodell trainieren und speichern
set.seed(123)
data <- data.frame(
  Feature1 = rnorm(100),
  Feature2 = rnorm(100),
  Target = rnorm(100)  # Zielvariable als kontinuierliche Variable (für Regression)
)

# Trainiere das lineare Regressionsmodell
lm_model <- train(Target ~ Feature1 + Feature2, data = data, method = "lm")

# Speichere das Modell als PMML-Datei
pmml_path <- "models/linear_model.pmml"
pmml_model <- pmml(lm_model$finalModel)
saveXML(pmml_model, file = pmml_path)

# Step 1a: Speichere den Datensatz als CSV-Datei
data_path <- "models/generated_data.csv"
write.csv(data, file = data_path, row.names = FALSE)  # Speichere den Datensatz ohne Zeilenummern

# Step 2: Surrogat-Grove erstellen
# Zielvariable aus den Daten entfernen, da das Surrogatmodell sie nicht benötigt
data_without_target <- data %>% select(-Target)

# Anzahl der Bäume, die für das Grove-Modell getestet werden
ntrees <- c(4, 8, 16, 32, 64, 128)

# Verwende xgrove, um das Surrogat-Grove-Modell zu erstellen
surrogate_grove <- xgrove(
  model = lm_model$finalModel,  # Das lineare Modell als Basis für das Surrogat
  data = data_without_target,
  ntrees = ntrees,
  shrink = 1,
  b.frac = 1,
  seed = 42
)

# Anzeige der Surrogatmodell-Ergebnisse
print(surrogate_grove)

# Optional: Plot zur Analyse der Komplexität des Grove-Modells im Vergleich zur Erklärbarkeit
plot(surrogate_grove, abs = "trees", ord = "upsilon")

# Bestätigung, dass der Datensatz gespeichert wurde
cat("Der Datensatz wurde unter 'models/generated_data.csv' gespeichert.\n")
