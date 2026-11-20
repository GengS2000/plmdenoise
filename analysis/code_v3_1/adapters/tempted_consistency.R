#!/usr/bin/env Rscript
# Run the authors' CRAN package `tempted` on one dataset and export the same
# two quantities the Gemelli adapter exports: the low-rank reconstruction on the
# transformed (CLR) scale, and the composition its inverse transform gives.
#
#   Rscript tempted_consistency.R --input counts.tsv.gz --clr-out clr.tsv.gz \
#       --comp-out comp.tsv.gz --diagnostics diag.json --rank 3 \
#       [--smooth 1e-6 --resolution 101 --maxiter 20 --epsilon 1e-4 --pseudo 1]
#
# Settings mirror the Gemelli defaults: CLR of (counts + pseudo), rank-1 SVD
# centring, the same smoothing, resolution, iteration cap and tolerance. The
# reconstruction is tempted::reconstruct(), i.e. the authors' own accessor.
suppressWarnings(suppressMessages(library(tempted)))

args <- commandArgs(trailingOnly = TRUE)
opt <- list(input = NA, clr_out = NA, comp_out = NA, diagnostics = NA, rank = 3,
            smooth = 1e-6, resolution = 101, maxiter = 20, epsilon = 1e-4, pseudo = 1)
i <- 1
while (i <= length(args)) {
  key <- gsub("-", "_", sub("^--", "", args[i]))
  if (!key %in% names(opt)) stop(paste("unknown option", args[i]))
  opt[[key]] <- args[i + 1]
  i <- i + 2
}

tab <- read.delim(gzfile(opt$input), header = TRUE, check.names = FALSE,
                  na.strings = c("NA", "NaN", ""))
if (!identical(colnames(tab)[1:3], c("subject", "visit", "time")))
  stop("input must start with subject/visit/time columns")
feature_names <- colnames(tab)[-(1:3)]
observed <- stats::complete.cases(tab[, -(1:3), drop = FALSE])
obs <- tab[observed, , drop = FALSE]

featuretable <- as.matrix(obs[, -(1:3), drop = FALSE])
rownames(featuretable) <- paste0("s", obs$subject, "_v", obs$visit)
# threshold = 1 keeps every feature: the benchmark scores one common taxon set.
datlist <- tempted::format_tempted(featuretable, obs$time, paste0("subject_", obs$subject),
                                   threshold = 1, pseudo = as.numeric(opt$pseudo),
                                   transform = "clr")
if (nrow(datlist[[1]]) - 1 != length(feature_names))
  stop("format_tempted dropped features; the benchmark needs all of them")

res_svd <- tempted::svd_centralize(datlist, r = 1)
started <- proc.time()
res <- tempted::tempted(res_svd$datlist, r = as.integer(opt$rank),
                        smooth = as.numeric(opt$smooth),
                        resolution = as.integer(opt$resolution),
                        maxiter = as.integer(opt$maxiter),
                        epsilon = as.numeric(opt$epsilon))
elapsed <- proc.time() - started
est <- tempted::reconstruct(res, res_svd = res_svd, r_reconstruct = as.integer(opt$rank))

clr_out <- matrix(NA_real_, nrow = nrow(tab), ncol = length(feature_names))
comp_out <- matrix(NA_real_, nrow = nrow(tab), ncol = length(feature_names))
# reconstruct() returns an unnamed list in subject order; the names come from
# the datlist it was built from.
subject_names <- names(res_svd$datlist)
if (length(subject_names) != length(est)) stop("reconstruct returned an unexpected length")
for (index in seq_along(est)) {
  name <- subject_names[index]
  subject <- sub("^subject_", "", name)
  block <- est[[index]]
  times <- block[1, ]
  values <- block[-1, , drop = FALSE]
  values <- values[feature_names, , drop = FALSE]     # by name, never by position
  rows <- which(tab$subject == as.integer(subject))
  for (k in seq_along(times)) {
    hit <- rows[which.min(abs(tab$time[rows] - times[k]))]
    clr_out[hit, ] <- values[, k]
    z <- values[, k] - max(values[, k])               # stable softmax
    comp_out[hit, ] <- exp(z) / sum(exp(z))
  }
}
colnames(clr_out) <- feature_names
colnames(comp_out) <- feature_names
for (path in c(opt$clr_out, opt$comp_out)) {
  values <- if (identical(path, opt$clr_out)) clr_out else comp_out
  con <- gzfile(path, "wt")
  write.table(cbind(tab[, 1:3], values), con, sep = "\t", quote = FALSE,
              row.names = FALSE, na = "NA")
  close(con)
}

lines <- c(
  "{",
  paste0('  "package": "tempted ', as.character(utils::packageVersion("tempted")), '",'),
  paste0('  "r_version": "', paste(R.version$major, R.version$minor, sep = "."), '",'),
  paste0('  "rank": ', as.integer(opt$rank), ','),
  paste0('  "smooth": ', as.numeric(opt$smooth), ','),
  paste0('  "resolution": ', as.integer(opt$resolution), ','),
  paste0('  "maxiter": ', as.integer(opt$maxiter), ','),
  paste0('  "epsilon": ', as.numeric(opt$epsilon), ','),
  paste0('  "pseudo_count": ', as.numeric(opt$pseudo), ','),
  paste0('  "lambda": [', paste(format(res$Lambda, digits = 10), collapse = ", "), '],'),
  paste0('  "features": ', length(feature_names), ','),
  paste0('  "subjects": ', length(datlist), ','),
  paste0('  "observed_samples": ', sum(observed), ','),
  paste0('  "elapsed_seconds": ', format(unname(elapsed["elapsed"]), digits = 10)),
  "}")
writeLines(lines, opt$diagnostics)
