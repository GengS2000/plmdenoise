# --check only inspects/loads installed packages; --install explicitly installs missing dependencies.
args <- commandArgs(trailingOnly = TRUE)
if (!length(args) || length(args) > 2L || !args[1] %in% c("--check", "--install")) stop("Use --check/--install [core|all|linda|maaslin2|glmmpql|zigmm]")
scope <- if (length(args) == 2L) args[2] else "all"
if (!scope %in% c("core", "maaslin2", "all", "linda", "glmmpql", "zigmm")) stop("Invalid scope")
args <- args[1]
cat("R:", R.version.string, "scope:", scope, "\n")
print(.libPaths())
lib <- Sys.getenv("R_LIBS_USER")
if (args == "--install" && !nzchar(lib)) stop("Set R_LIBS_USER to a writable server library")
if (nzchar(lib)) {
  if (args == "--install") dir.create(lib, recursive = TRUE, showWarnings = FALSE)
  .libPaths(c(lib, .libPaths()))
}
core <- c("MicrobiomeStat", "MASS", "nlme", "lme4", "lmerTest", "NBZIMM")
required <- switch(scope, core = core, maaslin2 = "Maaslin2", all = c(core, "Maaslin2"),
                   linda = c("MicrobiomeStat", "lme4", "lmerTest"),
                   glmmpql = c("MASS", "nlme"), zigmm = c("NBZIMM", "nlme"))
available <- function(pkg) requireNamespace(pkg, quietly = TRUE)
if (args == "--install") {
  options(repos = c(CRAN = "https://cloud.r-project.org"))
  cran <- c(setdiff(required, c("Maaslin2", "NBZIMM")),
            if ("NBZIMM" %in% required) "remotes", if ("Maaslin2" %in% required) "BiocManager")
  missing <- cran[!vapply(cran, available, logical(1))]
  if (length(missing)) install.packages(missing, lib = lib)
  if ("Maaslin2" %in% required && !available("Maaslin2")) BiocManager::install("Maaslin2", lib = lib, ask = FALSE, update = FALSE)
  if ("NBZIMM" %in% required && !available("NBZIMM")) {
    ref <- Sys.getenv("NBZIMM_REF", "master")
    remotes::install_github(paste0("nyiuab/NBZIMM@", ref), lib = lib, upgrade = "never", build_vignettes = FALSE)
  }
}
ok <- vapply(required, available, logical(1))
print(data.frame(package = required, available = unname(ok),
                 version = vapply(required, function(p) if (available(p)) as.character(packageVersion(p)) else NA_character_, character(1))))
if (!all(ok)) stop("Missing DA dependencies: ", paste(required[!ok], collapse = ", "))
if ("MicrobiomeStat" %in% required) stopifnot(all(c("feature.dat.type", "formula") %in% names(formals(MicrobiomeStat::linda))))
if ("NBZIMM" %in% required) stopifnot(all(c("fixed", "random", "zi_fixed") %in% names(formals(NBZIMM::lme.zig))))
if ("Maaslin2" %in% required) stopifnot(all(c("fixed_effects", "random_effects") %in% names(formals(Maaslin2::Maaslin2))))
cat("Dependency/API check passed. No simulation or DA fit performed.\n")
