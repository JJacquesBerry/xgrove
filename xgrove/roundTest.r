utils::globalVariables(c("left")) # resolves note on 'no visible binding for global variable 'left'' in group_by() in ln.123.

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

