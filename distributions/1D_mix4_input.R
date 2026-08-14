# ============================================================
# 1D_mix4_input.R
# Mixture of 5 with spike Beta(1200, 800):
# 0.1*U(0,1) + 0.2*Beta(2,5) + 0.2*Beta(1200,800)
#   + 0.3*TruncNorm(0.5, 0.1, [0.1, 0.9]) + 0.2*TruncNorm(0.7, 0.05, [0.3, 0.8])
# Truncated normals implemented with base R qnorm/pnorm (inverse-CDF
# method) -- no truncnorm package dependency.
# ============================================================

weights_mix4 <- c(0.1, 0.2, 0.2, 0.3, 0.2)

.tn_mean1  <- 0.5; .tn_sd1  <- 0.1; .tn_lo1 <- 0.1; .tn_hi1 <- 0.9
.tn_mean2  <- 0.7; .tn_sd2  <- 0.05; .tn_lo2 <- 0.3; .tn_hi2 <- 0.8

.rtruncnorm <- function(n, mean, sd, lower, upper) {
  u <- runif(n, pnorm(lower, mean, sd), pnorm(upper, mean, sd))
  qnorm(u, mean, sd)
}

.dtruncnorm <- function(x, mean, sd, lower, upper) {
  dens <- dnorm(x, mean, sd) / (pnorm(upper, mean, sd) - pnorm(lower, mean, sd))
  dens[x < lower | x > upper] <- 0
  return(dens)
}

extract_samples <- function(n) {
  comp    <- sample(1:5, size = n, replace = TRUE, prob = weights_mix4)
  samples <- numeric(n)
  samples[comp == 1] <- runif(sum(comp == 1), 0, 1)
  samples[comp == 2] <- rbeta(sum(comp == 2), 2, 5)
  samples[comp == 3] <- rbeta(sum(comp == 3), 1200, 800)
  samples[comp == 4] <- .rtruncnorm(sum(comp == 4), .tn_mean1, .tn_sd1, .tn_lo1, .tn_hi1)
  samples[comp == 5] <- .rtruncnorm(sum(comp == 5), .tn_mean2, .tn_sd2, .tn_lo2, .tn_hi2)
  return(matrix(samples, ncol = 1))
}

compute_pdf <- function(x) {
  weights_mix4[1] * dunif(x, 0, 1) +
    weights_mix4[2] * dbeta(x, 2, 5) +
    weights_mix4[3] * dbeta(x, 1200, 800) +
    weights_mix4[4] * .dtruncnorm(x, .tn_mean1, .tn_sd1, .tn_lo1, .tn_hi1) +
    weights_mix4[5] * .dtruncnorm(x, .tn_mean2, .tn_sd2, .tn_lo2, .tn_hi2)
}
