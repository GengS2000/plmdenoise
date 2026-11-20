#!/usr/bin/env Rscript
# MALLARD pilot: run the authors' toblerone.stan on one small benchmark dataset.
#
#   Rscript mallard_runner.R --input counts.tsv.gz --output comp.tsv.gz \
#       --diagnostics diag.json --stan-dir <MALLARD-Paper-Code/mdlm_toblerone> \
#       [--chains 4 --iter 2000 --warmup 1000 --adapt-delta 0.9 --max-treedepth 12 --seed 1]
#
# The Stan model is the authors' file, unchanged and compiled straight from the
# repository (mdlm_utils.stan is pulled in by its #include).  The data block is
# assembled exactly as in the authors' own mdlm_toblerone/simulation.R:
# random-walk DLM (FF = GG = I), m0 = 0, C0 = 100 I, lognormal(-1.5, 2) /
# lognormal(-1, 2) scale priors and LKJ(1) correlation priors.
#
# What is reported is the *biological* state theta (generated quantities), not
# the observation state eta: eta still carries the technical noise V.  Each
# posterior draw is mapped to a composition with the ILR inverse and the draws
# are averaged **on the composition scale**, never in ILR coordinates.
suppressWarnings(suppressMessages({
  library(rstan)
  library(compositions)
}))

args <- commandArgs(trailingOnly = TRUE)
opt <- list(input = NA, output = NA, diagnostics = NA, stan_dir = NA,
            chains = 4, iter = 2000, warmup = 1000, adapt_delta = 0.9,
            max_treedepth = 12, seed = 1, cores = 1)
i <- 1
while (i <= length(args)) {
  key <- gsub("-", "_", sub("^--", "", args[i]))
  if (!key %in% names(opt)) stop(paste("unknown option", args[i]))
  opt[[key]] <- args[i + 1]
  i <- i + 2
}
set.seed(as.integer(opt$seed))
rstan_options(auto_write = TRUE)
options(mc.cores = as.integer(opt$cores))

tab <- read.delim(gzfile(opt$input), header = TRUE, check.names = FALSE,
                  na.strings = c("NA", "NaN", ""))
if (!identical(colnames(tab)[1:3], c("subject", "visit", "time")))
  stop("input must start with subject/visit/time columns")
feature_names <- colnames(tab)[-(1:3)]
subjects <- sort(unique(tab$subject))
visits <- sort(unique(tab$visit))
r <- length(subjects)              # MALLARD's independent series = our subjects
TT <- length(visits)               # shared discrete time grid
N_species <- length(feature_names)
q <- N_species - 1
if (nrow(tab) != r * TT) stop("input is not a complete subject x visit grid")
if (N_species < 3) stop("MALLARD needs at least three taxa")

# Y[time, series, taxon]; Y_obs flags which (time, series) were sequenced.
# Stan cannot take NA in an integer array, so unobserved cells carry 0 and are
# excluded by Y_obs -- the authors' own mechanism (see simulation.R's Y.obs).
Y <- array(0L, dim = c(TT, r, N_species))
Y_obs <- matrix(0L, TT, r)
for (row in seq_len(nrow(tab))) {
  s <- match(tab$subject[row], subjects)
  t <- match(tab$visit[row], visits)
  counts <- as.numeric(tab[row, -(1:3)])
  if (!any(is.na(counts))) {
    Y[t, s, ] <- as.integer(round(counts))
    Y_obs[t, s] <- 1L
  }
}
if (sum(Y_obs) == 0) stop("no observed samples")

# ILR basis. The authors derive the contrast matrix from a phylogeny; the
# simulation has no tree, so the default orthonormal ILR basis is used and
# recorded. The likelihood is basis-invariant; the LKJ/lognormal priors on V
# and W are not, which the pilot report states.
contrast_matrix <- unclass(ilrBase(D = N_species))
dimnames(contrast_matrix) <- NULL
storage.mode(contrast_matrix) <- "double"

p <- q
stan_data <- list(
  N_timepoints_total = TT,
  N_timepoints_observed = TT,
  N_timepoints_sample = TT,
  r = r,
  N_species = N_species,
  p = p,
  TT_obs_ind = Y_obs,
  TT_rep_ind = matrix(0L, TT, r),
  TT_sample_ind = rep(1L, TT),
  Y = Y,
  Y_obs = Y_obs,
  contrast_matrix = contrast_matrix,
  d_F = 1,
  d_G = 1,
  FF = array(diag(p), dim = c(1, p, q)),
  GG = array(diag(p), dim = c(1, p, p)),
  m0 = matrix(0, r, p),
  C0 = 100 * diag(p),
  W_scale_mean_prior = rep(-1.5, p),
  W_scale_var_prior = rep(2, p),
  V_scale_mean_prior = rep(-1, q),
  V_scale_var_prior = rep(2, q),
  W_lkj_prior = 1,
  V_lkj_prior = 1)

model_code <- rstan::stanc_builder(file.path(opt$stan_dir, "toblerone.stan"),
                                   isystem = opt$stan_dir)
started <- proc.time()
model <- rstan::stan_model(model_code = model_code$model_code, verbose = FALSE)
fit <- rstan::sampling(model, data = stan_data,
                       chains = as.integer(opt$chains),
                       iter = as.integer(opt$iter),
                       warmup = as.integer(opt$warmup),
                       seed = as.integer(opt$seed),
                       control = list(adapt_delta = as.numeric(opt$adapt_delta),
                                      max_treedepth = as.integer(opt$max_treedepth)))
elapsed <- proc.time() - started

# theta: draws x N_timepoints_sample x r x p (biological state, ILR coordinates)
theta <- rstan::extract(fit, pars = "theta", permuted = TRUE)$theta
n_draws <- dim(theta)[1]
if (!identical(dim(theta)[2:4], as.integer(c(TT, r, p))))
  stop(sprintf("unexpected theta dimensions: got %s, expected %s",
               paste(dim(theta)[2:4], collapse = "x"), paste(c(TT, r, p), collapse = "x")))

# Posterior mean **on the composition scale**: average ilrInv(theta) over draws.
comp_sum <- array(0, dim = c(TT, r, N_species))
for (m in seq_len(n_draws)) {
  slice <- matrix(theta[m, , , ], nrow = TT * r, ncol = p)
  pis <- unclass(ilrInv(slice, V = contrast_matrix))
  comp_sum <- comp_sum + array(pis, dim = c(TT, r, N_species))
}
comp <- comp_sum / n_draws
# Cross-check against the Stan ILR_INV formula on the first draw.
check <- exp(contrast_matrix %*% theta[1, 1, 1, ])
check <- check / sum(check)
first <- unclass(ilrInv(matrix(theta[1, 1, 1, ], nrow = 1), V = contrast_matrix))
if (max(abs(as.numeric(first) - as.numeric(check))) > 1e-8)
  stop("compositions::ilrInv disagrees with the Stan ILR_INV formula")

out <- tab[, 1:3]
values <- matrix(NA_real_, nrow = nrow(tab), ncol = N_species)
for (row in seq_len(nrow(tab))) {
  s <- match(tab$subject[row], subjects)
  t <- match(tab$visit[row], visits)
  values[row, ] <- comp[t, s, ]
}
colnames(values) <- feature_names
con <- gzfile(opt$output, "wt")
write.table(cbind(out, values), con, sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")
close(con)

# --- diagnostics --------------------------------------------------------------
monitor <- rstan::summary(fit, pars = c("theta", "V", "W"))$summary
sampler <- rstan::get_sampler_params(fit, inc_warmup = FALSE)
divergences <- sum(sapply(sampler, function(x) sum(x[, "divergent__"])))
max_td <- max(sapply(sampler, function(x) max(x[, "treedepth__"])))
hit_td <- sum(sapply(sampler, function(x) sum(x[, "treedepth__"] >= as.integer(opt$max_treedepth))))
json_num <- function(x) if (is.finite(x)) format(x, digits = 10) else "null"
lines <- c(
  "{",
  paste0('  "rstan_version": "', as.character(utils::packageVersion("rstan")), '",'),
  paste0('  "r_version": "', paste(R.version$major, R.version$minor, sep = "."), '",'),
  paste0('  "chains": ', as.integer(opt$chains), ','),
  paste0('  "iter": ', as.integer(opt$iter), ','),
  paste0('  "warmup": ', as.integer(opt$warmup), ','),
  paste0('  "draws": ', n_draws, ','),
  paste0('  "adapt_delta": ', as.numeric(opt$adapt_delta), ','),
  paste0('  "max_treedepth": ', as.integer(opt$max_treedepth), ','),
  paste0('  "seed": ', as.integer(opt$seed), ','),
  paste0('  "rhat_max": ', json_num(max(monitor[, "Rhat"], na.rm = TRUE)), ','),
  paste0('  "rhat_frac_gt_1_01": ',
         json_num(mean(monitor[, "Rhat"] > 1.01, na.rm = TRUE)), ','),
  paste0('  "ess_bulk_min": ', json_num(min(monitor[, "n_eff"], na.rm = TRUE)), ','),
  paste0('  "ess_median": ', json_num(stats::median(monitor[, "n_eff"], na.rm = TRUE)), ','),
  paste0('  "divergences": ', divergences, ','),
  paste0('  "treedepth_max": ', max_td, ','),
  paste0('  "treedepth_saturated": ', hit_td, ','),
  paste0('  "n_species": ', N_species, ','),
  paste0('  "n_series": ', r, ','),
  paste0('  "n_timepoints": ', TT, ','),
  paste0('  "observed_samples": ', sum(Y_obs), ','),
  paste0('  "contrast_matrix": "compositions::ilrBase(D = N_species)",'),
  paste0('  "state": "theta (biological state), not eta",'),
  paste0('  "posterior_mean_scale": "composition",'),
  paste0('  "elapsed_seconds": ', format(unname(elapsed["elapsed"]), digits = 10), ','),
  paste0('  "cpu_seconds": ', format(unname(elapsed["user.self"] + elapsed["sys.self"]), digits = 10)),
  "}")
writeLines(lines, opt$diagnostics)
