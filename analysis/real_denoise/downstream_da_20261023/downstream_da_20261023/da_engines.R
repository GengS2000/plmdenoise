# DA_BATCH_WARNING_POLICY_V1
# DA adapters derived from the user's TUNE model choices; no legacy log10/NetCDF assumptions.
# All inputs are sample x taxon proportions. Truth is deliberately not accepted here.

da_capture <- function(expr) {
  warnings <- character()
  value <- tryCatch(withCallingHandlers(expr, warning = function(w) {
    warnings <<- c(warnings, conditionMessage(w))
    invokeRestart("muffleWarning")
  }), error = function(e) e)
  list(value = value, warnings = unique(warnings))
}

da_empty <- function(taxa) {
  data.frame(taxon = taxa, estimate = NA_real_, std_error = NA_real_,
             pvalue = NA_real_, fit_status = "no_result", fit_message = "",
             fit_engine = "unreported", iterations = NA_integer_,
             stringsAsFactors = FALSE)
}

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

da_align <- function(result, taxa) {
  if (anyDuplicated(result$taxon) || any(!result$taxon %in% taxa)) {
    stop("Unexpected or duplicated taxon identifiers returned by DA package")
  }
  out <- da_empty(taxa)
  at <- match(result$taxon, taxa)
  out[at, names(result)] <- result
  invalid <- !is.finite(out$pvalue) | out$pvalue < 0 | out$pvalue > 1 |
             !is.finite(out$estimate)
  out$fit_status[invalid & out$fit_status %in% c("ok", "warning")] <- "invalid_result"
  out$pvalue[invalid] <- NA_real_
  out
}

da_linda <- function(p, meta, out_dir, pseudocount) {
  captured <- da_capture(MicrobiomeStat::linda(
    feature.dat = t(p), meta.dat = meta,
    formula = paste("~", paste(c("Xcase", DA_NUISANCE), collapse=" + "), "+ (1 | participant)"),
    feature.dat.type = "proportion", prev.filter = 0, mean.abund.filter = 0,
    max.abund.filter = 0, is.winsor = FALSE, n.cores = 1, verbose = FALSE
  ))
  if (inherits(captured$value, "error")) stop(conditionMessage(captured$value))
  fit <- captured$value
  saveRDS(fit, file.path(out_dir, "linda_native.rds"))
  if (!"Xcase" %in% names(fit$output)) stop("LinDA did not return Xcase")
  tab <- fit$output[["Xcase"]]
  if (!all(c("log2FoldChange", "lfcSE", "pvalue") %in% colnames(tab))) stop("Unknown LinDA output schema")
  result <- data.frame(taxon = rownames(tab), estimate = tab$log2FoldChange,
                       std_error = tab$lfcSE, pvalue = tab$pvalue, fit_status = "ok", fit_message = "")
  result <- da_batch_results(result, colnames(p), captured$warnings, out_dir)
  result$fit_engine <- "MicrobiomeStat_linda_mixed"
  result
}

da_maaslin2 <- function(p, meta, out_dir, pseudocount) {
  # Same explicit log-proportion response as Gaussian glmmPQL; no package-side transform.
  response <- log2(p + pseudocount)
  safe <- sprintf("T%08d", seq_len(ncol(response)))
  original <- colnames(p)
  stopifnot(!anyDuplicated(original), !anyDuplicated(safe))
  write.csv(data.frame(alias=safe,taxon=original),file.path(out_dir,"feature_alias_map.csv"),row.names=FALSE)
  colnames(response) <- safe
  captured <- da_capture(Maaslin2::Maaslin2(
    input_data = as.data.frame(response), input_metadata = meta,
    output = file.path(out_dir, "maaslin2"),
    fixed_effects = c("Xcase", DA_NUISANCE), random_effects = "participant",
    reference = NULL,
    normalization = "NONE", transform = "NONE", analysis_method = "LM",
    min_abundance = -Inf, min_prevalence = 0, min_variance = 0,
    standardize = FALSE, cores = 1, plot_heatmap = FALSE, plot_scatter = FALSE
  ))
  if (inherits(captured$value, "error")) stop(conditionMessage(captured$value))
  path <- file.path(out_dir, "maaslin2", "all_results.tsv")
  tab <- read.delim(path, check.names = FALSE, stringsAsFactors = FALSE)
  required <- c("feature", "metadata", "value", "coef", "stderr", "pval")
  if (!all(required %in% names(tab))) stop("Unknown MaAsLin2 output schema")
  tab <- tab[tab$metadata == "Xcase", , drop = FALSE]
  index <- match(tab$feature, safe)
  if(anyNA(index) || anyDuplicated(index)) stop("Unknown or duplicate MaAsLin2 feature alias")
  tab$feature <- original[index]
  result <- data.frame(taxon = tab$feature, estimate = tab$coef,
                       std_error = tab$stderr, pvalue = tab$pval,
                       fit_status = rep("ok", nrow(tab)), fit_message = rep("", nrow(tab)))
  # Never use MaAsLin2's q-values across all covariates; Python applies BH to X only.
  result <- da_batch_results(result, colnames(p), captured$warnings, out_dir)
  result$fit_engine <- "Maaslin2_LM_random_intercept"
  result
}

da_lme_table <- function(fit, taxon, engine) {
  tab <- summary(fit)$tTable
  if (is.null(tab) || !"Xcase" %in% rownames(tab)) stop("Mixed model did not return Xcase")
  data.frame(taxon = taxon, estimate = tab["Xcase", "Value"],
             std_error = tab["Xcase", "Std.Error"], pvalue = tab["Xcase", "p-value"],
             fit_status = "ok", fit_message = "", fit_engine = engine,
             iterations = if (!is.null(fit$iter)) as.integer(fit$iter) else NA_integer_)
}

da_per_taxon <- function(p, meta, method, pseudocount) {
  result <- da_empty(colnames(p))
  for (j in seq_len(ncol(p))) {
    dat <- meta
    dat$y <- if (method == "glmmpql") log2(p[, j] + pseudocount) else asin(sqrt(p[, j]))
    engine <- if (method == "glmmpql") "MASS_glmmPQL_gaussian" else
      if (all(dat$y > 0)) "ZIGMM_boundary_pi0_nlme_ML" else "NBZIMM_lme.zig"
    result$fit_engine[j] <- engine
    if (length(unique(dat$y)) < 2) {
      result$fit_status[j] <- "constant_response"
      next
    }
    captured <- da_capture({
      fit <- if (method == "glmmpql") {
        MASS::glmmPQL(fixed = reformulate(c("Xcase", DA_NUISANCE), response="y"),
                      random = ~1 | participant, family = gaussian(link = "identity"),
                      data = dat, correlation = NULL, niter = 30, verbose = FALSE,
                      control = nlme::lmeControl(returnObject = FALSE))
      } else if (all(dat$y > 0)) {
        # Predeclared exact zero-mass boundary, not a fit-error fallback.
        # NBZIMM rejects no-zero responses; the ZIG likelihood then maximises at pi=0.
        nlme::lme(fixed = reformulate(c("Xcase", DA_NUISANCE), response="y"), random = ~1 | participant,
                  data = dat, method = "ML", correlation = NULL,
                  control = nlme::lmeControl(returnObject = FALSE))
      } else {
        # Same engine as mms(method='zig'), called per taxon to retain errors.
        # Arcsine sqrt preserves exact zeros and needs no library-size offset.
        NBZIMM::lme.zig(fixed = reformulate(c("Xcase", DA_NUISANCE), response="y"),
                        random = ~1 | participant, data = dat, correlation = NULL,
                        zi_fixed = ~1, zi_random = NULL, niter = 100,
                        epsilon = 1e-5, verbose = FALSE)
      }
      tab <- da_lme_table(fit, colnames(p)[j], engine)
      if (engine == "NBZIMM_lme.zig" &&
          ((!is.null(fit$iter) && fit$iter >= 100L) ||
           (!is.null(fit$zi.fit$converged) && !isTRUE(fit$zi.fit$converged)))) {
        tab$fit_status <- "nonconverged"
        tab$fit_message <- "ZIGMM iteration budget reached or zero-part GLM not converged"
        tab$pvalue <- NA_real_
      }
      if (engine == "ZIGMM_boundary_pi0_nlme_ML") {
        tab$fit_message <- "No zeros: predeclared zero-inflation probability=0 boundary"
      }
      tab
    })
    if (inherits(captured$value, "error")) {
      result$fit_status[j] <- "fit_error"
      result$fit_message[j] <- paste(conditionMessage(captured$value),
                                      paste(captured$warnings, collapse = " | "), sep = " | ")
    } else {
      result[j, ] <- da_apply_warnings(captured$value, captured$warnings)
    }
  }
  da_align(result, colnames(p))
}

