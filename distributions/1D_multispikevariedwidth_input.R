# ============================================================
# 1D_multispikevariedwidth_input.R
# Uniform background with 5 narrow spikes of varying widths
# at dyadic-adjacent locations (0.1, 0.3, 0.5, 0.7, 0.9).
# ============================================================

spike_centers_mvw    <- c(0.1, 0.3, 0.5, 0.7, 0.9)
spike_half_width_mvw <- c(0.08, 0.04, 0.0025, 0.01, 0.0005)
# wider spikes get extra weight so they read as taller, not flatter;
# weight pulled off the narrowest spike (5th) caps its height at 75
weights_mvw_raw      <- c(16, 39, 24, 16, 16, 9)      # background + 5 spikes
weights_mvw          <- weights_mvw_raw / sum(weights_mvw_raw)

extract_samples <- function(n) {
  comp    <- sample(1:6, size = n, replace = TRUE, prob = weights_mvw)
  samples <- numeric(n)
  samples[comp == 1] <- runif(sum(comp == 1), 0, 1)
  for (i in seq_along(spike_centers_mvw)) {
    idx <- (comp == (i + 1))
    if (sum(idx) > 0) {
      c_i <- spike_centers_mvw[i]
      hw_i <- spike_half_width_mvw[i]
      samples[idx] <- runif(sum(idx), c_i - hw_i, c_i + hw_i)
    }
  }
  return(matrix(samples, ncol = 1))
}

compute_pdf <- function(x) {
  y <- weights_mvw[1] * dunif(x, 0, 1)
  for (i in seq_along(spike_centers_mvw)) {
    c_i <- spike_centers_mvw[i]
    hw_i <- spike_half_width_mvw[i]
    y <- y + weights_mvw[i + 1] * dunif(x, c_i - hw_i, c_i + hw_i)
  }
  return(y)
}
