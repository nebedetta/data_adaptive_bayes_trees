# 3D_smooth1: smooth, near-symmetric, mildly-concentrated independent Beta
# product -- the "easy"/baseline case for median vs midpoint splitting.
#
# Rationale: this scenario has NO sharp features, NO strong off-center skew,
# and NO structure that midpoint splitting would handle poorly -- each
# marginal is a mild, roughly-symmetric Beta centered near 0.5, matching
# midpoint's fixed geometric split almost exactly. Median and midpoint
# splitting are expected to perform SIMILARLY here; this is the baseline
# against which the other two (sharper, more asymmetric) scenarios should
# show a growing median advantage.

d <- 3
shapes <- list(c(4, 4), c(5, 5), c(4.5, 4.5))

extract_samples <- function(n) {
  samples <- matrix(0, nrow = n, ncol = d)
  for (j in 1:d) {
    samples[, j] <- rbeta(n, shapes[[j]][1], shapes[[j]][2])
  }
  return(samples)
}

compute_pdf <- function(points) {
  density <- rep(1, nrow(points))
  for (j in 1:d) {
    density <- density * dbeta(points[, j], shapes[[j]][1], shapes[[j]][2])
  }
  return(density)
}
