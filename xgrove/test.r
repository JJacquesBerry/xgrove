library(randomForest)
library(gbm)
library(xgrove)
library(pmml)
library(pdp)

data(boston)

save_model_as_pmml <- function(model, file_path) {
  if (!requireNamespace("pmml", quietly = TRUE)) {
    stop("Das pmml-Paket ist nicht installiert. Bitte installiere es mit install.packages('pmml').")
  }
  
  pmml_model <- pmml::pmml(model)
  cat(as.character(pmml_model), file = file_path)
  message("Das Modell wurde erfolgreich als PMML-Datei gespeichert unter: ", file_path)
}

set.seed(42)
rf <- randomForest(cmedv ~ ., data = boston)
data <- boston[, -which(names(boston) == "cmedv")]
ntrees <- c(4, 8, 16, 32, 64, 128)

xg <- xgrove(rf, data, ntrees)
print(xg)
plot(xg)

pmml_file_path <- "models/analyzed_model.pmml"
save_model_as_pmml(rf, pmml_file_path)
