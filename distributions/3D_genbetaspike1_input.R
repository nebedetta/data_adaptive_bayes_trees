source('distributions/functions_for_mixtures.R')

# 3D_genbetaspike1: a single, sharp, off-center, off-dyadic generalized Beta
# spike (Olkin & Liu 2003 gamma-ratio construction, same family as
# 5D_genbetaspike1), with genuinely different per-dimension sharpness and
# location -- no two dimensions share a mean or a sharpness parameter, and
# none of the three per-dimension means sit near a shallow binary split
# point (0.5, 0.25, 0.75, ...), so a fixed midpoint tree can never cleanly
# isolate this peak by accident the way it would if a component location
# happened to coincide with a dyadic split point.
#
# Construction:
#   g0 ~ Gamma(a0, b0)
#   gj ~ Gamma(aj, bj),  j = 1..3   (independent of g0 and each other)
#   xj = gj / (gj + g0)
# Target means (0.71, 0.29, 0.79) verified off-dyadic through depth 9.
# With a0=150 and aj=(110, 88, 143), the joint density peaks around ~5800 at
# its mode (std ~0.02-0.028 per axis) vs. ~0 at the cube center.

d  <- 3
a0 <- 150
b0 <- 1

target_means <- c(0.71, 0.29, 0.79)
aj_vec <- c(110, 88, 143)  # 110, 110*0.8, 110*1.3 -- different sharpness per dim
bj_vec <- aj_vec / (target_means / (1 - target_means) * a0 / b0)

shapes <- lapply(1:d, function(j) c(aj_vec[j], bj_vec[j]))

extract_samples <- function(n) {
  g0 <- rgamma(n, a0, b0)
  samples <- matrix(0, nrow = n, ncol = d)
  for (j in 1:d) {
    gj <- rgamma(n, shapes[[j]][1], shapes[[j]][2])
    samples[, j] <- gj / (gj + g0)
  }
  return(samples)
}

compute_pdf <- function(points) {
  lambdas <- sapply(1:d, function(j) shapes[[j]][2] / b0)
  a_list  <- sapply(1:d, function(j) shapes[[j]][1])
  a_sum   <- a0 + sum(a_list)
  logB    <- sum(lgamma(a_list)) + lgamma(a0) - lgamma(a_sum)

  log_num   <- rep(0, nrow(points))
  ratio_sum <- rep(0, nrow(points))
  for (j in 1:d) {
    xj  <- points[, j]
    aj_ <- a_list[j]
    lam <- lambdas[j]
    log_num   <- log_num + aj_ * log(lam) + (aj_ - 1) * log(xj) - (aj_ + 1) * log(1 - xj)
    ratio_sum <- ratio_sum + lam * xj / (1 - xj)
  }
  log_den <- a_sum * log(1 + ratio_sum)

  return(exp(log_num - log_den - logB))
}
