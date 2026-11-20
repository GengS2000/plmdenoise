#!/usr/bin/env Rscript
# Run the authors' microTensor on one benchmark dataset.
#
#   Rscript microtensor_runner.R --input counts.tsv.gz --output comp.tsv.gz \
#       --diagnostics diag.json --rank 3 [--weighted TRUE --ortho-m FALSE ...]
#
# The fit is microTensor::microTensor() unchanged.  The only thing added here is
# the export of p-hat, computed exactly as in the authors' helpers_evaluation.R
# (stable softmax of get_Yhat() over the feature mode).  Nothing is re-fit or
# re-implemented; the time mode is the shared discrete visit grid, which the
# caller has already verified.
suppressWarnings(suppressMessages(library(microTensor)))

args <- commandArgs(trailingOnly = TRUE)
opt <- list(input = NA, output = NA, diagnostics = NA, rank = 3,
            weighted = "TRUE", ortho_m = "FALSE", nn_t = "FALSE",
            maxit = 5000, abs_tol = 1e-4, rel_tol = 1e-4,
            L_init = 3, gamma = 2, init = "ctf", seed = 1)
i <- 1
while (i <= length(args)) {
  key <- sub("^--", "", args[i])
  key <- gsub("-", "_", key)
  if (!key %in% names(opt)) stop(paste("unknown option", args[i]))
  opt[[key]] <- args[i + 1]
  i <- i + 2
}
as_logical <- function(x) toupper(as.character(x)) %in% c("TRUE", "T", "1", "YES")
rank <- as.integer(opt$rank)
set.seed(as.integer(opt$seed))

# ---- read the long table -----------------------------------------------------
tab <- read.delim(gzfile(opt$input), header = TRUE, check.names = FALSE,
                  na.strings = c("NA", "NaN", ""))
key_cols <- c("subject", "visit", "time")
if (!identical(colnames(tab)[1:3], key_cols))
  stop("input must start with subject/visit/time columns")
feature_names <- colnames(tab)[-(1:3)]
subjects <- sort(unique(tab$subject))
visits <- sort(unique(tab$visit))
n_feature <- length(feature_names)
n_subject <- length(subjects)
n_visit <- length(visits)
if (nrow(tab) != n_subject * n_visit)
  stop("input is not a complete subject x visit grid")

# microTensor's X is feature x subject x time; a visit nobody sampled is NA
# across every feature, which is how the authors' own simulation marks it.
X <- array(NA_real_, dim = c(n_feature, n_subject, n_visit),
           dimnames = list(feature_names, as.character(subjects), as.character(visits)))
row_subject <- match(tab$subject, subjects)
row_visit <- match(tab$visit, visits)
counts <- as.matrix(tab[, -(1:3), drop = FALSE])
for (row in seq_len(nrow(tab)))
  X[, row_subject[row], row_visit[row]] <- counts[row, ]

observed <- apply(X, c(2, 3), function(x) !all(is.na(x)))
if (any(apply(X, c(2, 3), function(x) any(is.na(x)) && !all(is.na(x)))))
  stop("a visit has some but not all taxa missing; microTensor masks whole visits")

started <- proc.time()
fit <- microTensor::microTensor(
  X = X, R = rank,
  weighted = as_logical(opt$weighted),
  ortho_m = as_logical(opt$ortho_m),
  nn_t = as_logical(opt$nn_t),
  control = list(L_init = as.numeric(opt$L_init),
                 gamma = as.numeric(opt$gamma),
                 init = as.character(opt$init),
                 abs_tol = as.numeric(opt$abs_tol),
                 rel_tol = as.numeric(opt$rel_tol),
                 maxit = as.integer(opt$maxit),
                 verbose = FALSE))
elapsed <- proc.time() - started

# ---- author-defined relative composition -------------------------------------
# helpers_evaluation.R: phat = softmax over the feature mode of the linear
# predictor Yhat, computed with the same max-subtraction for stability.
Yhat <- microTensor::get_Yhat(fit)
phat <- apply(Yhat, c(2, 3), function(x) {
  x <- x - max(x)
  exp(x) / sum(exp(x))
})
if (!identical(dim(phat), dim(X))) stop("phat does not have the input tensor shape")

out <- tab[, 1:3]
values <- matrix(NA_real_, nrow = nrow(tab), ncol = n_feature)
for (row in seq_len(nrow(tab)))
  values[row, ] <- phat[, row_subject[row], row_visit[row]]
colnames(values) <- feature_names
out <- cbind(out, values)
con <- gzfile(opt$output, "wt")
write.table(out, con, sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")
close(con)

json_number_list <- function(x) paste0("[", paste(format(x, digits = 10), collapse = ", "), "]")
lines <- c(
  "{",
  paste0('  "package_version": "', as.character(utils::packageVersion("microTensor")), '",'),
  paste0('  "r_version": "', paste(R.version$major, R.version$minor, sep = "."), '",'),
  paste0('  "rank": ', rank, ','),
  paste0('  "weighted": ', tolower(as.character(as_logical(opt$weighted))), ','),
  paste0('  "ortho_m": ', tolower(as.character(as_logical(opt$ortho_m))), ','),
  paste0('  "nn_t": ', tolower(as.character(as_logical(opt$nn_t))), ','),
  paste0('  "init": "', opt$init, '",'),
  paste0('  "maxit": ', as.integer(opt$maxit), ','),
  paste0('  "seed": ', as.integer(opt$seed), ','),
  paste0('  "lambda": ', json_number_list(fit$lambda), ','),
  paste0('  "objective": ', json_number_list(fit$fitting$obj), ','),
  paste0('  "steps": ', json_number_list(fit$fitting$steps), ','),
  paste0('  "convergence": ', json_number_list(fit$fitting$convergence), ','),
  paste0('  "converged": ', tolower(as.character(all(fit$fitting$convergence == 0))), ','),
  paste0('  "observed_visits": ', sum(observed), ','),
  paste0('  "input_shape": [', paste(dim(X), collapse = ", "), '],'),
  paste0('  "elapsed_seconds": ', format(unname(elapsed["elapsed"]), digits = 10), ','),
  paste0('  "cpu_seconds": ', format(unname(elapsed["user.self"] + elapsed["sys.self"]), digits = 10)),
  "}")
writeLines(lines, opt$diagnostics)
