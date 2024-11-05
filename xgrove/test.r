library(randomForest)
library(pmml)

set.seed(42)
num_rows <- 80

data <- data.frame(
  Alter = sample(18:70, num_rows, replace = TRUE),
  Gewicht = sample(50:100, num_rows, replace = TRUE),
  Einkommen = sample(20000:100000, num_rows, replace = TRUE),
  Geschlecht = sample(c("Männlich", "Weiblich"), num_rows, replace = TRUE),
  Kategorie = sample(c("A", "B", "C"), num_rows, replace = TRUE),
  Zielwert = rnorm(num_rows, mean = 50000, sd = 15000)
)

data$Geschlecht <- as.factor(data$Geschlecht)
data$Kategorie <- as.factor(data$Kategorie)

write.csv(data, "models/generated_data.csv", row.names = FALSE)

set.seed(42)
rf <- randomForest(Zielwert ~ ., data = data)

pmml_file_path <- "models/analyzed_model.pmml"

save_model_as_pmml <- function(model, file_path) {
  pmml_model <- pmml(model)
  
  if (is.null(pmml_model)) {
    stop("Fehler beim Erstellen des PMML-Modells.")
  }
  
  saveXML(pmml_model, file = file_path)
}

tryCatch({
  save_model_as_pmml(rf, pmml_file_path)
  print("Model gespeichert als PMML.")
}, error = function(e) {
  cat("Fehler beim Speichern des Modells:", e$message, "\n")
})
