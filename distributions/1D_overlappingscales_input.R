# ============================================================
# 1D_overlappingscales_input.R
# Overlapping structures of different scales:
# 0.1*U(0,1) + 0.3*U(0.25,0.5) + 0.4*Beta_[0.25,0.5](2,2) + 0.2*Beta(4000,6000)
# Sharp spike (Beta(4000,6000), std ~0.005) centered at x=0.4 --
# INSIDE the U(0.25,0.5)/Beta_[0.25,0.5](2,2) support, unlike
# 1D_spikemix's spike at x=0.6 (outside that support).
# ============================================================

weights_overlap <- c(0.1, 0.3, 0.4, 0.2)

extract_samples <- function(n) {
  comp    <- sample(1:4, size = n, replace = TRUE, prob = weights_overlap)
  samples <- numeric(n)
  samples[comp == 1] <- runif(sum(comp == 1), 0, 1)
  samples[comp == 2] <- runif(sum(comp == 2), 0.25, 0.5)
  samples[comp == 3] <- 0.25 + 0.25 * rbeta(sum(comp == 3), 2, 2)
  samples[comp == 4] <- rbeta(sum(comp == 4), 4000, 6000)
  return(matrix(samples, ncol = 1))
}

compute_pdf <- function(x) {
  weights_overlap[1] * dunif(x, 0, 1) +
    weights_overlap[2] * dunif(x, 0.25, 0.5) +
    weights_overlap[3] * (dbeta((x - 0.25) / 0.25, 2, 2) / 0.25) +
    weights_overlap[4] * dbeta(x, 4000, 6000)
}
