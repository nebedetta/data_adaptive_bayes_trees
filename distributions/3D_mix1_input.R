source('distributions/functions_for_mixtures.R')

# 3D_mix1: two generalized-Beta modes at different scales (broad vs sharp),
# well-separated, off-dyadic locations, equal weight.
#
# Rationale: no single fixed splitting depth is simultaneously right for
# both components -- the broad component (std ~0.09-0.13) is already
# well-resolved at shallow depth, while the sharp component (std ~0.02-0.026,
# peak ~5760 vs the broad component's ~48) needs much deeper, well-targeted
# splits to resolve. Both components' per-dimension means are off-dyadic and
# distinct from each other, so midpoint splitting can't accidentally land on
# either mode's location.
#
# Construction: same generalized Beta (Olkin & Liu 2003 gamma-ratio) family
# as 3D_genbetaspike1, mixed 50/50:
#   broad:  a0=8,   aj=(6,6,6)         -> std ~0.09-0.13, peak ~48   (mode ~(0.21,0.46,0.62))
#   sharp:  a0=150, aj=(110,110,110)   -> std ~0.02-0.026, peak ~5760 (mode ~(0.79,0.71,0.29))

d <- 3

gen_beta_component <- function(a0, aj_vec, target_means) {
  bj_vec <- aj_vec / (target_means / (1 - target_means) * a0)
  list(a0 = a0, b0 = 1, aj_vec = aj_vec, bj_vec = bj_vec)
}

comp_broad <- gen_beta_component(a0 = 8,   aj_vec = c(6, 6, 6),       target_means = c(0.21, 0.46, 0.62))
comp_sharp <- gen_beta_component(a0 = 150, aj_vec = c(110, 110, 110), target_means = c(0.79, 0.71, 0.29))

gen_beta_sample_nd <- function(n, comp) {
  g0 <- rgamma(n, comp$a0, comp$b0)
  samples <- matrix(0, nrow = n, ncol = d)
  for (j in 1:d) {
    gj <- rgamma(n, comp$aj_vec[j], comp$bj_vec[j])
    samples[, j] <- gj / (gj + g0)
  }
  return(samples)
}

gen_beta_density_nd <- function(points, comp) {
  a0 <- comp$a0; b0 <- comp$b0
  aj_vec <- comp$aj_vec; bj_vec <- comp$bj_vec
  lambda_vec <- bj_vec / b0
  a_sum  <- a0 + sum(aj_vec)
  logB   <- sum(lgamma(aj_vec)) + lgamma(a0) - lgamma(a_sum)

  log_num   <- rep(0, nrow(points))
  ratio_sum <- rep(0, nrow(points))
  for (j in 1:d) {
    xj  <- points[, j]
    aj_ <- aj_vec[j]
    lam <- lambda_vec[j]
    log_num   <- log_num + aj_ * log(lam) + (aj_ - 1) * log(xj) - (aj_ + 1) * log(1 - xj)
    ratio_sum <- ratio_sum + lam * xj / (1 - xj)
  }
  log_den <- a_sum * log(1 + ratio_sum)
  return(exp(log_num - log_den - logB))
}

component_sample <- list(
  function(n = 1) gen_beta_sample_nd(n, comp_broad),
  function(n = 1) gen_beta_sample_nd(n, comp_sharp)
)
component_pdf <- c(
  function(points) gen_beta_density_nd(points, comp_broad),
  function(points) gen_beta_density_nd(points, comp_sharp)
)
weights <- c(0.5, 0.5)

extract_samples <- function(n) {
  return(sample_from_mixture(component_sample, weights, n))
}

compute_pdf <- function(points) {
  return(pdf_for_mixture(component_pdf, weights, points))
}
