# Batch package warnings do not identify the taxon that emitted them.
# Keep native finite results, and record that convergence is not established.
da_apply_warnings <- function(result, warnings, scope = c("taxon", "run")) {
  scope <- match.arg(scope)
  if (!length(warnings)) return(result)
  message <- paste(warnings, collapse = " | ")
  if (scope == "run") message <- paste0("[run-level; taxon attribution unavailable] ", message)
  result$fit_message <- paste(result$fit_message, message, sep = " | ")
  bad <- any(grepl("converg|iteration limit|false convergence", warnings, ignore.case = TRUE))
  eligible <- result$fit_status == "ok"
  result$fit_status[eligible] <- if (bad && scope == "taxon") "nonconverged" else "warning"
  if (bad && scope == "taxon") result$pvalue[eligible] <- NA_real_
  result
}

da_batch_results <- function(result, taxa, warnings, out_dir) {
  # This file precedes alignment/validation/warning annotation. MaAsLin2 also
  # retains its original all_results.tsv. No p-value is recomputed here.
  write.csv(result, file.path(out_dir, "package_results_raw.csv"), row.names = FALSE, na = "NA")
  warnings <- unique(warnings)
  diagnostics <- data.frame(
    scope = rep("run", length(warnings)),
    taxon = rep(NA_character_, length(warnings)),
    message = warnings, stringsAsFactors = FALSE)
  write.csv(diagnostics, file.path(out_dir, "package_warnings.csv"), row.names = FALSE, na = "NA")
  result <- da_apply_warnings(da_align(result, taxa), warnings, scope = "run")
  result$warning_scope <- if (length(warnings)) "run_unattributed" else "none_observed"
  result$convergence_assessment <- "not_available_per_taxon"
  result
}
