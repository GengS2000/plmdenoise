# Dependency/API check only. No data generation or model fitting.
cat(R.version.string, "\n")
print(.libPaths())
suppressPackageStartupMessages(library(microTensor))
stopifnot(is.function(microTensor::microTensor))
print(packageVersion("microTensor"))
cat("microTensor dependency/API check passed. No fit performed.\n")
