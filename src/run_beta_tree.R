# run_betatree.R
#
# Fits BetaTree on all distributions × sample sizes × iterations and
# saves the estimated density at the shared LHS grid points.
#
# Raw output (per iteration):
#   {SIM_OUTPUT_PATH}/{prefix}/{prefix}_n{n}_iter{i}_BetaTree.csv.gz
#   single column "density", length = N_EVAL
#   mirrors: {SIM_OUTPUT_PATH}/{prefix}/{prefix}_n{n}_iter{i}_KDE_scott.npz
#
# Metric computation is done separately in Python (betatree_metrics.ipynb)
# using f.distance_metric, exactly as for KDE/DPM.
#
# Usage:
#   Rscript run_betatree.R

# install.packages("devtools")
# devtools::install_github("zq00/BetaTree")

library(BetaTree)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

REPO_ROOT      <- "."
SIM_OUTPUT_PATH <- file.path(REPO_ROOT, "output", "baselines", "sim")
DENSITY_FOLDER  <- file.path(REPO_ROOT, "distributions", "true_density")
SAMPLES_FOLDER  <- file.path(REPO_ROOT, "distributions", "samples")

PREFIXES     <- c(
  "2D_genbeta1",
  "2D_genbeta3",
  "2D_genbeta4",
  "2D_genbeta5",
  "2D_mix13", 
  "2D_mix14"
)

SAMPLE_SIZES <- c(5000)
N_ITER       <- 200
N_EVAL       <- 250000

ALPHA        <- 0.1
METHOD       <- "weighted_bonferroni"
BOUNDED      <- FALSE

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

evaluate_betatree <- function(hist_mat, points, d) {
  n_eval      <- nrow(points)
  density_out <- numeric(n_eval)
  lower_cols  <- 1:d
  upper_cols  <- (d + 1):(2 * d)
  den_col     <- 2 * d + 1
  
  for (r in seq_len(nrow(hist_mat))) {
    lo  <- hist_mat[r, lower_cols]
    hi  <- hist_mat[r, upper_cols]
    den <- hist_mat[r, den_col]
    inside <- rep(TRUE, n_eval)
    for (j in seq_len(d)) {
      inside <- inside & (points[, j] > lo[j]) & (points[, j] <= hi[j])
    }
    density_out[inside] <- den
  }
  density_out
}

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

dir.create(SIM_OUTPUT_PATH, recursive = TRUE, showWarnings = FALSE)

total_runs <- length(PREFIXES) * length(SAMPLE_SIZES) * N_ITER
done       <- 0
t_start    <- proc.time()["elapsed"]

for (PREFIX in PREFIXES) {
  
  cat(sprintf("\n── %s ────────────────────────────────\n", PREFIX))
  
  sim_prefix_dir <- file.path(SIM_OUTPUT_PATH, PREFIX)
  dir.create(sim_prefix_dir, recursive = TRUE, showWarnings = FALSE)
  
  # load grid
  grid_path <- file.path(file.path(REPO_ROOT, "distributions"), "grid_points_2D.csv.gz")
  if (!file.exists(grid_path)) {
    message("  [SKIP] grid not found: ", grid_path)
    done <- done + length(SAMPLE_SIZES) * N_ITER
    next
  }
  grid_df <- read.csv(grid_path)
  d       <- ncol(grid_df)
  points  <- as.matrix(grid_df)
  
  for (N in SAMPLE_SIZES) {
    for (i in seq(0, N_ITER - 1)) {
      
      out_path <- file.path(
        sim_prefix_dir,
        sprintf("%s_n%d_iter%d_BetaTree.csv.gz", PREFIX, N, i)
      )
      
      # skip if already done
      if (file.exists(out_path)) {
        done <- done + 1
        next
      }
      
      sample_path <- file.path(
        SAMPLES_FOLDER, PREFIX,
        sprintf("%s_n%d_iter%d.csv", PREFIX, N, i)
      )
      if (!file.exists(sample_path)) {
        message(sprintf("  [SKIP] sample not found: %s", sample_path))
        done <- done + 1
        next
      }
      
      X <- as.matrix(read.csv(sample_path))
      
      tryCatch({
        hist_mat <- BuildHist(X, alpha = ALPHA, method = METHOD,
                              bounded = BOUNDED, plot = FALSE)
        est_den  <- evaluate_betatree(hist_mat, points, d)
        
        gz_con <- gzfile(out_path, "w")
        write.csv(data.frame(density = est_den), gz_con, row.names = FALSE)
        close(gz_con)
        
      }, error = function(e) {
        message(sprintf("  [ERROR] n=%d iter=%d : %s", N, i,
                        conditionMessage(e)))
      })
      
      done <- done + 1
      if (done %% 200 == 0) {
        elapsed <- proc.time()["elapsed"] - t_start
        eta     <- (total_runs - done) / (done / max(elapsed, 1e-6))
        message(sprintf("[%d/%d] %.0fs elapsed | ETA %.0fs (~%.1fmin)",
                        done, total_runs, elapsed, eta, eta / 60))
      }
    }
  }
}

message(sprintf("\nDone. %d runs processed.", done))
