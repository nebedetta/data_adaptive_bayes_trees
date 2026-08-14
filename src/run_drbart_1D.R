# run_drbart_1D.R
#
# Fits DR-BART in an unconditional density estimation setting by using an
# (almost) constant covariate and writes the estimated density on a supplied
# evaluation grid.
#
# DR-BART is a conditional density regression method (Orlandi & Murray,
# https://github.com/vittorioorlandi/drbart). To obtain an unconditional
# density estimate, we fit the model using a single covariate with a small
# amount of random jitter around a constant value. The jitter is included for
# numerical stability, as the current implementation can be unstable when the
# covariate is exactly constant.
#
# The response is standardized (centered and divided by its sample standard
# deviation) prior to fitting, and the evaluation grid is standardized using
# the same transformation. Posterior predictive densities are transformed back
# to the original scale via the appropriate Jacobian correction.
#
# The model uses variance = "ux", matching the DR-BART package defaults.
#
# Meant to be called once per (distribution, n, iteration); see
# run_drbart() in src/sim_run_baselines_1D_drbart.py, which shells out to this
# script via subprocess (no rpy2 / in-process R bridge).
#
# Usage:
#   Rscript run_drbart_1D.R <y_csv_path> <grid_csv_path> <out_csv_path> \
#       [nburn] [nsim] [nthin] [seed]
#
#   y_csv_path    : CSV containing the sample y values
#   grid_csv_path : CSV containing the evaluation grid
#   out_csv_path  : output CSV containing the estimated density
#   nburn         : burn-in iterations (default 10000)
#   nsim          : retained posterior draws (default 1000)
#   nthin         : thinning interval (default 10)
#   seed          : RNG seed for DR-BART fitting (default 0)


suppressMessages(library(drbart))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript run_drbart_1D.R <y_csv_path> <grid_csv_path> <out_csv_path> [nburn] [nsim]")
}

y_path    <- args[1]
grid_path <- args[2]
out_path  <- args[3]

nburn  <- if (length(args) >= 4) as.integer(args[4]) else 10000
nsim   <- if (length(args) >= 5) as.integer(args[5]) else 1000
nthin  <- if (length(args) >= 6) as.integer(args[6]) else 10
seed   <- if (length(args) >= 7) as.integer(args[7]) else 0

y_raw     <- read.csv(y_path)[[1]]
ygrid_raw <- read.csv(grid_path)[[1]]

# Standardize response for DR-BART fitting
y_mean <- mean(y_raw)
y_sd   <- sd(y_raw)

y     <- (y_raw - y_mean) / y_sd
ygrid <- (ygrid_raw - y_mean) / y_sd

set.seed(seed)
x <- matrix(runif(length(y), 0.999, 1.001), ncol = 1)

mean_file <- tempfile(fileext = ".txt")
prec_file <- tempfile(fileext = ".txt")

t0 <- proc.time()[["elapsed"]]
fit <- drbart(
  y, x,
  nburn = nburn,
  nsim = nsim,
  nthin = nthin,
  variance = "ux",
  mean_file = mean_file,
  prec_file = prec_file,
  printevery = nburn + nsim * nthin
)
t_sample <- proc.time()[["elapsed"]] - t0

t0 <- proc.time()[["elapsed"]]
preds <- predict(
  fit,
  xpred = matrix(1),
  ygrid = ygrid,
  type = "density",
  n_cores = 4
)
t_predict <- proc.time()[["elapsed"]] - t0

cat(sprintf("run_drbart_1D.R timing: sampling=%.1fs  predict=%.1fs  (n=%d, n_eval=%d, nburn=%d, nsim=%d, nthin=%d)\n",
            t_sample, t_predict, length(y), length(ygrid), nburn, nsim, nthin))

# posterior mean density on standardized scale
est_den_z <- rowMeans(preds$preds[1, , ])

# transform density back to original y-scale
est_den <- est_den_z / y_sd

write.csv(
  data.frame(density = est_den),
  out_path,
  row.names = FALSE
)

files <- c(mean_file, prec_file)
files <- files[file.exists(files)]
if (length(files) > 0) {
  invisible(file.remove(files))
}